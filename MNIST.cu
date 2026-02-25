#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <chrono>
#include <random>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <thread>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#ifndef __CUDACC__
#define __syncthreads()
#endif


uint32_t swap_endian(uint32_t val) {
	return ((val << 24) & 0xFF000000) |
		((val << 8) & 0x00FF0000) |
		((val >> 8) & 0x0000FF00) |
		((val >> 24) & 0x000000FF);
}

void print_image(std::vector<uint8_t>& image_data, uint32_t num_rows, uint32_t num_cols) {
	for (uint32_t r = 0; r < num_rows; ++r) {
		for (uint32_t c = 0; c < num_cols; ++c) {
			uint32_t pixel = image_data[r * num_cols + c];

			if (pixel > 128) std::cout << "@@";
			else if (pixel > 50) std::cout << "..";
			else std::cout << "  ";

		}
		std::cout << std::endl;
	}

	return;
}


//Read images from dataset
std::tuple<uint32_t, uint32_t, uint32_t>  read_mnist_images(const std::string& filename, std::vector<std::vector<uint8_t>>& images) {
	std::ifstream file(filename, std::ios::binary);

	if (!file.is_open()) {
		std::cout << "Caught Error when open file" << std::endl;
		exit(EXIT_FAILURE);
	}

	images.clear();

	uint32_t magic_number = 0;
	uint32_t num_images = 0;
	uint32_t num_rows = 0;
	uint32_t num_cols = 0;

	file.read(reinterpret_cast<char*>(&magic_number), 4);
	file.read(reinterpret_cast<char*>(&num_images), 4);
	file.read(reinterpret_cast<char*>(&num_rows), 4);
	file.read(reinterpret_cast<char*>(&num_cols), 4);

	magic_number = swap_endian(magic_number);
	num_images = swap_endian(num_images);
	num_rows = swap_endian(num_rows);
	num_cols = swap_endian(num_cols);

	std::cout << "==== MNIST INFO ====" << std::endl;
	std::cout << "Magic Number: " << magic_number << std::endl;
	std::cout << "Image Quantity: " << num_images << std::endl;
	std::cout << "Image Size: " << num_rows << "x" << num_cols << std::endl;

	uint32_t total_pixels = num_rows * num_cols;

	std::vector<uint8_t> image_data(total_pixels);

	while (file.read(reinterpret_cast<char*>(image_data.data()), total_pixels))
		images.push_back(image_data);

	return std::tie(num_images, num_rows, num_cols);
}

//read labels from dataset
void read_labels_file(const std::string filename, std::vector<int> &labels) {
	std::ifstream file(filename, std::ios::binary);

	if (!file.is_open()) {
		std::cout << "Caught Error when open file" << std::endl;
		exit(EXIT_FAILURE);
	}

	labels.clear();

	uint32_t magic_number = 0;
	uint32_t num_labels = 0;

	file.read(reinterpret_cast<char*>(&magic_number), 4);
	file.read(reinterpret_cast<char*>(&num_labels), 4);
	
	magic_number = swap_endian(magic_number);
	num_labels = swap_endian(num_labels);
	
	std::cout << magic_number << std::endl;
	std::cout << num_labels << std::endl;

	uint8_t label_data;
	while (file.read(reinterpret_cast<char*>(&label_data), 1))
		labels.push_back(static_cast<int>(label_data));
}

const std::string img_test = "t10k-images.idx3-ubyte";
const std::string label_test = "t10k-labels.idx1-ubyte";
const std::string img_file = "train-images.idx3-ubyte";
const std::string label_file = "train-labels.idx1-ubyte";
std::vector<std::vector<uint8_t>> images, images_test;
std::vector<int> labels, labels_test;
uint32_t num_images, num_rows, num_cols;


__global__ void vec_add(float* a, float* b, float* c, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size) {
		c[idx] = a[idx] + b[idx];
	}
}


//A[m, k] x B[k, n] = C[m, n]
#define TILE_SIZE 16
__global__ void matrix_mul(float* A, float* B, float* C, int M, int K, int N) {
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	float sum = 0.0f;
	for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
		int k_A = tile * TILE_SIZE + tx;
		if (row < M && k_A < K) As[ty][tx] = A[row * K + k_A];
		else As[ty][tx] = 0.0f;

		int k_B = tile * TILE_SIZE + ty;
		if (k_B < K && col < N) Bs[ty][tx] = B[k_B * N + col];
		else Bs[ty][tx] = 0.0f;

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k)
			sum += As[ty][k] * Bs[k][tx];

		__syncthreads();
	}

	if (row < M && col < N) {
		C[row * N + col] = sum;
	}
}

// A[m, k] x B[n, k]^T = C[m, n]
__global__ void matrix_mul_transB(float* A, float* B, float* C, int M, int K, int N) {
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	float sum = 0.0f;
	for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
		int k_A = tile * TILE_SIZE + tx;
		if (row < M && k_A < K) As[ty][tx] = A[row * K + k_A];
		else As[ty][tx] = 0.0f;

		int k_B = tile * TILE_SIZE + ty;
		if (col < N && k_B < K) Bs[ty][tx] = B[col * K + k_B]; // B[col, k_B]
		else Bs[ty][tx] = 0.0f;

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k)
			sum += As[ty][k] * Bs[k][tx];

		__syncthreads();
	}

	if (row < M && col < N)
		C[row * N + col] = sum;
}

// A[k, m]^T x B[k, n] = C[m, n]
__global__ void matrix_mul_transA(float* A, float* B, float* C, int M, int K, int N) {
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	float sum = 0.0f;
	for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
		int k_A = tile * TILE_SIZE + tx;
		if (k_A < K && row < M) As[ty][tx] = A[k_A * M + row]; // A[k_A, row]
		else As[ty][tx] = 0.0f;

		int k_B = tile * TILE_SIZE + ty;
		if (k_B < K && col < N) Bs[ty][tx] = B[k_B * N + col];
		else Bs[ty][tx] = 0.0f;

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k)
			sum += As[ty][k] * Bs[k][tx];

		__syncthreads();
	}

	if (row < M && col < N)
		C[row * N + col] = sum;
}

__global__ void bias_addition(float* x, float* b, int size_x, int size_b) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size_x) {
		x[idx] += b[idx % size_b];
	}
}

__global__ void ReLU(float* x, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		x[idx] = fmaxf(0.0f, x[idx]);
	}
}

__global__ void softmax(float* x, int batch_size, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < batch_size) {
		float max_val = x[idx * size];
		for (int i = 1; i < size; ++i)
			max_val = fmaxf(max_val, x[idx * size + i]);

		float sum = 0.0f;
		for (int i = 0; i < size; ++i) {
			x[idx * size + i] = expf(x[idx * size + i] - max_val);
			sum += x[idx * size + i];
		}

		for (int i = 0; i < size; ++i) {
			x[idx * size + i] /= sum;
		}
	}
}

__global__ void zero_grad(float* grad, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		grad[idx] = 0.0f;
	}
}

/*
	Backpropagation:
	z(i, l) = a(i, l - 1) * w(j, l) + b(l)
	dL/dw(i, j, l) = dL/dz(i, l) * dz(i, l)/dw(i, j, l) = dL/dz(i, l) * a(i, l - 1)
	
	a(i, l + 1) = f(z(i, l)) (f = activation function)

	dL/dz(i, l) = dL/da(i, l + 1) * da(i, l + 1)/dz(i, l) = dL/da(i, l + 1) * f'(z(i, l))
	
	dL/da(i, l + 1) = sum(dL/dz(k, l + 1) * dz(k, l + 1)/da(i, l + 1)) = sum(dL/dz(k, + 1) * w(i, k, l + 1))
*/

__global__ void ReLU_backward(float* dZ1, float* x, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		dZ1[idx] = (x[idx] > 0.0f) ? dZ1[idx] : 0.0f;
	}
}

__global__ void softmax_cross_entropy_backward(float* dZ2, float* probs, int* labels, int batch_size, int num_classes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < batch_size * num_classes) {
		int b = idx / num_classes;
		int c = idx % num_classes;
		dZ2[idx] = (probs[idx] - (c == labels[b] ? 1.0f : 0.0f)) / batch_size;
	}
}

__global__ void bias_backward(float* db, float* dZ, int batch_size, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		for (int i = 0; i < batch_size; ++i)
			db[idx] += dZ[i * size + idx];
	}
}

__global__ void adam_update(float* W, float* dW, float* m, float* v, float lr, float beta1, float beta2, float epsilon, int t, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		float grad = dW[idx];
		m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
		v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

		float m_hat = m[idx] / (1.0f - powf(beta1, t));
		float v_hat = v[idx] / (1.0f - powf(beta2, t));

		W[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
	}
}

const unsigned int batch_size = 64;

/*
	Basic MLP Network:
	28 x 28 pixels image -> 784-d vector
	W1[784, 1024] -> feature extractor with ReLU activation function
	W2[1024, 10] -> compute probabilities of which number is handwritten 
*/

class NeuralNetwork {
public:
	void initializeMatrix(float*& a, int rows, int cols) {
		float scale = sqrtf(2.0f / rows);
		cudaMallocManaged(&a, rows * cols * sizeof(float));
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> real_dist(0.0f, 2.0f);

		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j) {
				a[i * cols + j] = real_dist(rng) * scale;
			}
	}

	void initializeBias(float*& b, int size) {
		cudaMallocManaged(&b, size * sizeof(float));
		std::fill(b, b + size, 0.0f);
	}

	NeuralNetwork(int input_size1, int input_size2, int output_size) {
		this->t = 1;
		this->learning_rate = 0.001f;
		this->beta1 = 0.9f;
		this->beta2 = 0.999f;
		this->epsilon = 1e-8f;

		cudaMallocManaged(&this->m_w1, input_size1 * input_size2 * sizeof(float));
		cudaMallocManaged(&this->v_w1, input_size1 * input_size2 * sizeof(float));
		cudaMallocManaged(&this->m_b1, input_size2 * sizeof(float));
		cudaMallocManaged(&this->v_b1, input_size2 * sizeof(float));
		cudaMallocManaged(&this->m_w2, input_size2 * output_size * sizeof(float));
		cudaMallocManaged(&this->v_w2, input_size2 * output_size * sizeof(float));
		cudaMallocManaged(&this->m_b2, output_size * sizeof(float));
		cudaMallocManaged(&this->v_b2, output_size * sizeof(float));

		cudaMemset(this->m_w1, 0, input_size1 * input_size2 * sizeof(float));
		cudaMemset(this->v_w1, 0, input_size1 * input_size2 * sizeof(float));
		cudaMemset(this->m_b1, 0, input_size2 * sizeof(float));
		cudaMemset(this->v_b1, 0, input_size2 * sizeof(float));
		cudaMemset(this->m_w2, 0, input_size2 * output_size * sizeof(float));
		cudaMemset(this->v_w2, 0, input_size2 * output_size * sizeof(float));
		cudaMemset(this->m_b2, 0, output_size * sizeof(float));
		cudaMemset(this->v_b2, 0, output_size * sizeof(float));

		this->w1_rows = input_size1;
		this->w1_cols = this->bias1_size = this->w2_rows = input_size2;
		this->w2_cols = this->bias2_size = output_size;

		initializeMatrix(this->weights1, input_size1, input_size2);
		initializeBias(this->bias1, input_size2);
		initializeMatrix(this->weights2, input_size2, output_size);
		initializeBias(this->bias2, output_size);

		cudaMallocManaged(&this->grad_weights1, input_size1 * input_size2 * sizeof(float));
		cudaMallocManaged(&this->grad_weights2, input_size2 * output_size * sizeof(float));
		cudaMallocManaged(&this->grad_bias1, input_size2 * sizeof(float));
		cudaMallocManaged(&this->grad_bias2, output_size * sizeof(float));

		cudaMallocManaged(&this->cache_out1, batch_size * this->w1_cols * sizeof(float));
		cudaMallocManaged(&this->cache_out2, batch_size * this->w2_cols * sizeof(float));
		cudaMallocManaged(&this->dZ1, batch_size * this->w1_cols * sizeof(float));
		cudaMallocManaged(&this->dZ2, batch_size * this->w2_cols * sizeof(float));
	}

	~NeuralNetwork() {
		cudaFree(weights1);  cudaFree(weights2);
		cudaFree(bias1);     cudaFree(bias2);
		cudaFree(grad_weights1); cudaFree(grad_weights2);
		cudaFree(grad_bias1);    cudaFree(grad_bias2);
		cudaFree(cache_out1);    cudaFree(cache_out2);
		cudaFree(dZ1);           cudaFree(dZ2);
	}

	float* forward(float* x, int current_batch_size) {
		dim3 blockDim(TILE_SIZE, TILE_SIZE);

		dim3 grid1((this->w1_cols + TILE_SIZE - 1) / TILE_SIZE, (current_batch_size + TILE_SIZE - 1) / TILE_SIZE);
		matrix_mul << <grid1, blockDim >> > (x, this->weights1, this->cache_out1, current_batch_size, this->w1_rows, this->w1_cols);
		cudaDeviceSynchronize();

		bias_addition << <(current_batch_size * this->bias1_size + 255) / 256, 256 >> > (this->cache_out1, this->bias1, current_batch_size * this->bias1_size, this->bias1_size);
		cudaDeviceSynchronize();

		ReLU << <(current_batch_size * this->bias1_size + 255) / 256, 256 >> > (this->cache_out1, current_batch_size * this->bias1_size);
		cudaDeviceSynchronize();

		dim3 grid2((this->w2_cols + TILE_SIZE - 1) / TILE_SIZE, (current_batch_size + TILE_SIZE - 1) / TILE_SIZE);
		matrix_mul << <grid2, blockDim >> > (this->cache_out1, this->weights2, this->cache_out2, current_batch_size, this->w2_rows, this->w2_cols);
		cudaDeviceSynchronize();

		bias_addition << <(current_batch_size * this->bias2_size + 255) / 256, 256 >> > (this->cache_out2, this->bias2, current_batch_size * this->bias2_size, this->bias2_size);
		cudaDeviceSynchronize();

		softmax << <(current_batch_size + 255) / 256, 256 >> > (this->cache_out2, current_batch_size, this->bias2_size);
		cudaDeviceSynchronize();

		return this->cache_out2;
	}

	void backward(float* x, int* labels, int current_batch_size) {
		dim3 blockDim(TILE_SIZE, TILE_SIZE);

		// 1. Compute dZ2
		softmax_cross_entropy_backward << <(current_batch_size * this->w2_cols + 255) / 256, 256 >> > (this->dZ2, this->cache_out2, labels, current_batch_size, this->w2_cols);
		cudaDeviceSynchronize();

		// 2. Compute dW2 = A1^T x dZ2
		dim3 grid_dW2((this->w2_cols + TILE_SIZE - 1) / TILE_SIZE, (this->w2_rows + TILE_SIZE - 1) / TILE_SIZE);
		matrix_mul_transA << <grid_dW2, blockDim >> > (this->cache_out1, this->dZ2, this->grad_weights2, this->w2_rows, current_batch_size, this->w2_cols);

		// 3. Compute db2
		bias_backward << <(this->bias2_size + 255) / 256, 256 >> > (this->grad_bias2, this->dZ2, current_batch_size, this->bias2_size);

		// 4. Compute dZ1 = dZ2 x W2^T
		dim3 grid_dZ1((this->w1_cols + TILE_SIZE - 1) / TILE_SIZE, (current_batch_size + TILE_SIZE - 1) / TILE_SIZE);
		matrix_mul_transB << <grid_dZ1, blockDim >> > (this->dZ2, this->weights2, this->dZ1, current_batch_size, this->w2_cols, this->w1_cols);
		cudaDeviceSynchronize();

		// 5. Backprop through
		ReLU_backward << <(current_batch_size * this->w1_cols + 255) / 256, 256 >> > (this->dZ1, this->cache_out1, current_batch_size * this->w1_cols);
		cudaDeviceSynchronize();

		// 6. Compute dW1 = X^T x dZ1
		dim3 grid_dW1((this->w1_cols + TILE_SIZE - 1) / TILE_SIZE, (this->w1_rows + TILE_SIZE - 1) / TILE_SIZE);
		matrix_mul_transA << <grid_dW1, blockDim >> > (x, this->dZ1, this->grad_weights1, this->w1_rows, current_batch_size, this->w1_cols);
		cudaDeviceSynchronize();

		// 7. Compute db1
		bias_backward << <(this->w1_cols + 255) / 256, 256 >> > (this->grad_bias1, this->dZ1, current_batch_size, this->w1_cols);
		cudaDeviceSynchronize();

		// 8. Update Weights and Biases
		adam_update<<<(this->w1_rows * this->w1_cols + 255) / 256, 256>>> (this->weights1, this->grad_weights1, this->m_w1, this->v_w1, this->learning_rate, this->beta1, this->beta2, this->epsilon, this->t, this->w1_rows * this->w1_cols);
		adam_update<<<(this->bias1_size + 255) / 256, 256>>> (this->bias1, this->grad_bias1, this->m_b1, this->v_b1, this->learning_rate, this->beta1, this->beta2, this->epsilon, this->t, this->bias1_size);
		adam_update<<<(this->w2_rows * this->w2_cols + 255) / 256, 256>>> (this->weights2, this->grad_weights2, this->m_w2, this->v_w2, this->learning_rate, this->beta1, this->beta2, this->epsilon, this->t, this->w2_rows * this->w2_cols);
		adam_update<<<(this->bias2_size + 255) / 256, 256>>> (this->bias2, this->grad_bias2, this->m_b2, this->v_b2, this->learning_rate, this->beta1, this->beta2, this->epsilon, this->t, this->bias2_size);
		cudaDeviceSynchronize();

		this->t += 1;
	}

	void set_zero_grad() {
		zero_grad << <(this->w1_rows * this->w1_cols + 255) / 256, 256 >> > (this->grad_weights1, this->w1_rows * this->w1_cols);
		zero_grad << <(this->w2_rows * this->w2_cols + 255) / 256, 256 >> > (this->grad_weights2, this->w2_rows * this->w2_cols);
		zero_grad << <(this->bias1_size + 255) / 256, 256 >> > (this->grad_bias1, this->bias1_size);
		zero_grad << <(this->bias2_size + 255) / 256, 256 >> > (this->grad_bias2, this->bias2_size);
	}

private:
	int t;
	float beta1;
	float beta2;
	float epsilon;

	float* m_w1;
	float* v_w1;
	float* m_b1;
	float* v_b1;
	float* m_w2;
	float* v_w2;
	float* m_b2;
	float* v_b2;


	float learning_rate;
	float* weights1;
	float* weights2;
	float* bias1;
	float* bias2;
	float* cache_out1;
	float* cache_out2;
	float* grad_weights1;
	float* grad_weights2;
	float* grad_bias1;
	float* grad_bias2;
	float* dZ1;
	float* dZ2;
	size_t w1_rows;
	size_t w1_cols;
	size_t bias1_size;
	size_t w2_rows;
	size_t w2_cols;
	size_t bias2_size;
};


double compute_cross_entropy_loss(float* probs, int* labels, int current_batch_size, int num_classes) {
	double batch_loss = 0.0;

	for (int i = 0; i < current_batch_size; ++i) {
		int true_class = labels[i];

		float p = probs[i * num_classes + true_class];

		p = fmaxf(p, 1e-7f);

		batch_loss -= logf(p);
	}

	return batch_loss;
}

void Training(NeuralNetwork& Net) {
	int num_epochs = 30;
	int num_classes = 10;
	int input_size = num_rows * num_cols;

	std::vector<int> index(num_images);
	std::iota(index.begin(), index.end(), 0);
	std::mt19937 rng(42);

	float* d_data;
	int* d_labels;
	cudaMallocManaged(&d_data, batch_size * input_size * sizeof(float));
	cudaMallocManaged(&d_labels, batch_size * sizeof(int));

	for (int epoch = 0; epoch < num_epochs; ++epoch) {
		std::shuffle(index.begin(), index.end(), rng);

		double total_loss = 0.0;
		int num_batches = 0;

		for (int i = 0; i < num_images; i += batch_size) {
			int current_batch_size = std::min(batch_size, num_images - i);

			for (int j = 0; j < current_batch_size; ++j) {
				int original_idx = index[i + j];

				for (size_t k = 0; k < input_size; ++k) {
					d_data[j * input_size + k] = float(images[original_idx][k]) / 255.0f;
				}
				d_labels[j] = labels[original_idx];
			}

			Net.set_zero_grad();

			float* probs = Net.forward(d_data, current_batch_size);

			cudaDeviceSynchronize();

			double batch_loss = compute_cross_entropy_loss(probs, d_labels, current_batch_size, num_classes);
			total_loss += batch_loss;

			Net.backward(d_data, d_labels, current_batch_size);

			num_batches++;

			if (num_batches % 50 == 0) {
				std::cout << "\rTraining Epoch " << epoch + 1 << "/" << num_epochs
					<< " | Progress: " << (int) (float(i + current_batch_size) / num_images * 100) << "%"
					<< std::flush;
			}
		}

		double avg_loss = total_loss / num_images;

		std::cout << "\rEpoch " << epoch + 1 << "/" << num_epochs
			<< " Completed | Average Loss: " << avg_loss << "                     \n";
	}

	cudaFree(d_data);
	cudaFree(d_labels);
}

void Evaluating(NeuralNetwork& Net) {
	int input_size = num_rows * num_cols;

	float* d_data;
	int* d_labels;
	cudaMallocManaged(&d_data, batch_size * input_size * sizeof(float));
	cudaMallocManaged(&d_labels, batch_size * sizeof(int));


	int acc = 0;
	for (int i = 0; i < num_images; i += batch_size) {
		int current_batch_size = std::min(batch_size, num_images - i);

		for (int j = 0; j < current_batch_size; ++j) {
			for (size_t k = 0; k < input_size; ++k) {
				d_data[j * input_size + k] = float(images[i + j][k]) / 255.0f;
			}
			d_labels[j] = labels[i + j];
		}

		float* probs = Net.forward(d_data, current_batch_size);

		cudaDeviceSynchronize();

		for (int j = 0; j < current_batch_size; ++j) {
			int label = 0;
			for (size_t k = 1; k < 10; ++k)
				if (probs[j * 10 + k] > probs[j * 10 + label])
					label = k;

			acc += (label == d_labels[j]);
		}

		std::cout << "\rEvaluating"
		<< " | Progress: " << (int) (float(i + current_batch_size) / num_images * 100) << "%"
		<< std::flush;

	}
	std::cout << "\n\n\nAccuracy: " << float(acc) / num_images * 100.0f << "%" << std::flush;
	cudaFree(d_data);
	cudaFree(d_labels);
}


int main(int argc, char* argv[]) {
	std::tie(num_images, num_rows, num_cols) = read_mnist_images(img_file, images);
	read_labels_file(label_file, labels);
	std::cout << "====== EXAMPLE ======\n";
	for (int i = 0; i < 5; ++i) {
		print_image(images[i], num_rows, num_cols);
		std::cout << "Label: " << labels[i] << '\n';
	}

	NeuralNetwork Net(784, 1024, 10);

	Training(Net);

	std::tie(num_images, num_rows, num_cols) = read_mnist_images(img_test, images_test);
	read_labels_file(label_test, labels_test);

	Evaluating(Net);

}