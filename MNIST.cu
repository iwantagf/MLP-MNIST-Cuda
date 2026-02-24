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

const std::string img_file = "t10k-images.idx3-ubyte";
const std::string label_file = "t10k-labels.idx1-ubyte";
std::vector<std::vector<uint8_t>> images;
std::vector<int> labels;
uint32_t num_images, num_rows, num_cols;


__global__ void vec_add(float* a, float* b, float* c, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size) {
		c[idx] = a[idx] + b[idx];
	}
}

#define TILE_SIZE 16
__global__ void matrix_mul(float* A, float* B, float* C, int M, int K, int N) {
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int row = bx * TILE_SIZE + tx;
	int col = by * TILE_SIZE + ty;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	float sum = 0.0f;

	for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
		if (row < M && tile * TILE_SIZE + ty < K)
			As[tx][ty] = A[row * K + tile * TILE_SIZE + ty];
		else
			As[tx][ty] = 0.0f;

		if (tile * TILE_SIZE + tx < K && col < N)
			Bs[tx][ty] = B[(tile * TILE_SIZE + tx) * N + col];
		else
			Bs[tx][ty] = 0.0f;

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k)
			sum += As[tx][k] * Bs[k][ty];

		__syncthreads();
	}

	if (row < M && col < N) {
		C[row * N + col] = sum;
	}
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

__global__ void ReLU_backward(float* grad, float *x, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size) {
		grad[idx] *= (x[idx] > 0);
	}
}

__global__ void softmax_backward(float* grad, float* x, int batch_size, int input_size, int output_size) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.x;
	
	if (row < input_size && col < output_size) {
		
	}
}

float CrossEntropyLoss(float* a) {
	if (a == NULL) {
		std::cout << "Null pointer detected";
		exit(EXIT_FAILURE);
	}

	float ans = 0.0f;
	for (int i = 0; i < 10; ++i)
		ans += -std::log(a[i] + 1e-7f);

	return ans;
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

	float* forward(float* x, int batch_size) {
		dim3 blockDim(TILE_SIZE, TILE_SIZE);

		cudaMallocManaged(&this->cache_out1, batch_size * this->w1_cols * sizeof(float));

		matrix_mul<<<dim3((batch_size + TILE_SIZE - 1) / TILE_SIZE, (this->w1_cols + TILE_SIZE - 1) / TILE_SIZE), blockDim>>>(x, this->weights1, this->cache_out1, batch_size, this->w1_rows, this->w1_cols);
		cudaDeviceSynchronize();

		bias_addition<<<(batch_size * this->bias1_size + 255) / 256, 256>>>(this->cache_out1, this->bias1, batch_size * this->bias1_size, this->bias1_size);
		cudaDeviceSynchronize();

		ReLU<<<(batch_size * this->bias1_size + 255) / 256, 256>>>(this->cache_out1, batch_size * this->bias1_size);
		cudaDeviceSynchronize();

		cudaMallocManaged(&this->cache_out2, batch_size * this->w2_cols * sizeof(float));

		matrix_mul<<<dim3((batch_size + TILE_SIZE - 1) / TILE_SIZE, (this->w2_cols + TILE_SIZE - 1) / TILE_SIZE), blockDim>>> (this->cache_out1, this->weights2, this->cache_out2, batch_size, this->w2_rows, this->w2_cols);
		cudaDeviceSynchronize();

		bias_addition<<<(batch_size * this->bias2_size + 255) / 256, 256>>>(this->cache_out2, this->bias2, batch_size * this->bias2_size, this->bias2_size);
		cudaDeviceSynchronize();

		softmax<<<(batch_size + 255) / 256, 256>>>(this->cache_out2, batch_size, this->bias2_size);
		cudaDeviceSynchronize();

		
		return this->cache_out2;
	}

	void backward(float* output, float* x) {
		
	}

	void set_zero_grad() {
		zero_grad<<<(this->w1_rows * this->w1_cols + 255) / 256, 256>>>(this->grad_weights1, this->w1_rows * this->w1_cols);
		zero_grad<<<(this->w2_rows * this->w2_cols + 255) / 256, 256>>>(this->grad_weights2, this->w2_rows * this->w2_cols);
		zero_grad<<<(this->bias1_size + 255) / 256, 256>>> (this->grad_bias1, this->bias1_size);
		zero_grad<<<(this->bias2_size + 255) / 256, 256>>> (this->grad_bias2, this->bias2_size);
	}

	NeuralNetwork(int input_size1, int input_size2, int output_size) {
		this->learning_rate = 0.01f;
		initializeMatrix(this->weights1, input_size1, input_size2);
		initializeBias(this->bias1, input_size2);
		initializeMatrix(this->weights2, input_size2, output_size);
		initializeBias(this->bias2, output_size);

		cudaMallocManaged(&this->grad_weights1, input_size1 * input_size2 * sizeof(float));
		cudaMallocManaged(&this->grad_weights2, input_size2 * output_size * sizeof(float));
		cudaMallocManaged(&this->grad_bias1, input_size2 * sizeof(float));
		cudaMallocManaged(&this->grad_bias2, output_size * sizeof(float));

		this->w1_rows = input_size1;
		this->w1_cols = this->bias1_size = this->w2_rows = input_size2;
		this->w2_cols = this->bias2_size = output_size;
	}

	~NeuralNetwork() {
		cudaFree(weights1);
		cudaFree(weights2);
		cudaFree(bias1);
		cudaFree(bias2);
		cudaFree(grad_weights1);
		cudaFree(grad_weights2);
		cudaFree(grad_bias1);
		cudaFree(grad_bias2);
		cudaFree(cache_out1);
		cudaFree(cache_out2);
	}

private:
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
	size_t w1_rows;
	size_t w1_cols;
	size_t bias1_size;
	size_t w2_rows;
	size_t w2_cols;
	size_t bias2_size;
};

void Training(NeuralNetwork& Net) {
	int num_epochs = 50;

	for (int epoch = 0; epoch < num_epochs; ++epoch) {
		Net.set_zero_grad();
	}
}

void Evaluating(NeuralNetwork& Net) {

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
	Evaluating(Net);

}