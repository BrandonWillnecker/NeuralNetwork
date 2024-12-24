// Michael Nielsen - Neural Networks and Deep Learning
// http://neuralnetworksanddeeplearning.com/
// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://github.com/unexploredtest/neural-networks-and-deep-learning
// Reimplemented in c++ for neural network projects, now includes simulated annleaing and adam optimization
// This is extnded upon to form the Chromosome object for the NEAT algorithm in some other projects

#pragma once
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <chrono>
#include <mutex>
#include <execution>
#include <algorithm>

class Timer
{
public:
	Timer(float* duration) {
		this->duration = duration;
		start_time_point = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		stop();
	}

	void stop() {
		using namespace std::chrono;
		auto end_time_point = high_resolution_clock::now();
		auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(start_time_point).time_since_epoch().count();
		auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time_point).time_since_epoch().count();
		auto duration = (float)(end - start);
		*(this->duration) = duration;
	}

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
	float* duration;

};

class Random
{
public:
	//Default constructs the RNG with current time as the seed
	Random() :
		m_seed((uint32_t)std::chrono::system_clock::now().time_since_epoch().count())
	{}

	//Constructs the RNG with the given seed
	Random(uint32_t seed) :
		m_seed(seed)
	{}

	//Generates a random float in [0,1]
	float next_float() {
		m_seed = pcg_hash(m_seed);
		return (float)m_seed / (float)UINT32_MAX;
	}

	//Generates a random float in [-1,1]
	float next_ufloat() {
		return 2.0f * next_float() - 1.0f;
	}

	//Generates a random uint32
	uint32_t next_uint() {
		m_seed = pcg_hash(m_seed);
		return m_seed;
	}

	//Sets the seed of the RNG
	void seed(uint32_t seed) {
		m_seed = seed;
	}

	//Shuffle given generic vector
	template<typename T>
	void shuffle(std::vector<T>& arr) {
		size_t size = arr.size();
		for (size_t i = 0; i < size; i++) {
			std::swap(arr[i], arr[next_uint() % size]);
		}
	}

private:
	uint32_t m_seed = 0;

	uint32_t pcg_hash(uint32_t input) {
		uint32_t state = input * 747796405u + 2891336453u;
		uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
		return (word >> 22u) ^ word;
	}
};

struct Vector
{
	Vector() {}

	Vector(const std::initializer_list<float>& init_list) 
		:rows(init_list.size()), values(new float[init_list.size()])
	{
		std::memcpy(values, init_list.begin(), init_list.size() * sizeof(float));
	}

	Vector(size_t rows)
		:rows(rows), values(new float[rows])
	{
		std::memset(values, 0, rows * sizeof(float));
	}

	Vector(size_t rows, Random& random) :
		rows(rows), values(new float[rows])
	{
		for (size_t row = 0; row < rows; row++)
			values[row] = random.next_ufloat();
	}

	Vector(const Vector& other) 
		:rows(other.rows), values(new float[other.rows])
	{
		std::memcpy(values, other.values, rows*sizeof(float));
	}

	Vector(Vector&& other) noexcept
		:rows(other.rows), values(std::exchange(other.values,nullptr))
	{
	}

	~Vector() {
		delete[] values;
	}

	Vector& operator=(const Vector& other) = delete;

	Vector& operator=(Vector&& other) = delete;

	void print() const {
		std::cout << '[';
		for (size_t i = 0; i < rows; i++)
			std::cout << values[i] << ", ";
		std::cout << "]\n";
	}
	
	void set_zero() {
		std::memset(values, 0, rows * sizeof(float));
	}

	void add_row() {
		float* values_new = new float[rows + 1];
		std::memcpy(values_new, values, rows * sizeof(float));
		values_new[rows] = 0.0f;
		delete[] values;
		values = values_new;
		rows++;
	}

	std::string to_npy_str() const {
		std::string str = "np.matrix([";
		for (size_t row = 0; row < rows; row++) {
			str += "[" + std::to_string(values[row]) + "], ";
		}
		return str + "])";
	}

	size_t rows = 0;
	float* values = nullptr;
};

struct Matrix
{
	Matrix() {}

	Matrix(size_t rows, size_t cols)
		:rows(rows), cols(cols), values(new float[rows*cols])
	{
		std::memset(values, 0, rows * cols * sizeof(float));
	}

	Matrix(size_t rows, size_t cols, Random& random)
		:rows(rows), cols(cols), values(new float[rows * cols])
	{
		for (size_t i = 0; i < rows * cols; i++) {
			values[i] = random.next_ufloat();
		}
	}

	Matrix(const std::initializer_list<float>& init_list, size_t rows, size_t cols)
	:rows(rows), cols(cols), values(new float[rows*cols])
	{
		std::memcpy(values, init_list.begin(), rows * cols * sizeof(float));
	}

	Matrix(const Matrix& other)
		:rows(other.rows), cols(other.cols), values(new float[other.rows*other.cols])
	{
		std::memcpy(values, other.values, rows * cols * sizeof(float));
	}

	Matrix(Matrix&& other) noexcept
		:rows(other.rows), cols(other.cols), values(std::exchange(other.values, nullptr))
	{
	}

	~Matrix() {
		delete[] values;
	}

	Matrix& operator=(const Matrix& other) {
		if (rows != other.rows || cols != other.cols) {
			delete[] values;
			rows = other.rows;
			cols = other.cols;
			values = new float[rows * cols];
		}	
		std::memcpy(values, other.values, rows * cols* sizeof(float));
		return *this;
	}

	Matrix& operator=(Matrix&& other) noexcept {
		rows = other.rows;
		cols = other.cols;
		std::swap(values, other.values);
		return *this;
	}

	void print() const {
		size_t index = 0;
		for (size_t row = 0; row < rows; row++) {
			std::cout << '[';
			for (size_t col = 0; col < cols; col++) {
				std::cout << values[index] << ", ";
				index++;
			}
			std::cout << "]\n";
		}
	}
	
	void set_zero() {
		std::memset(values, 0, rows * cols * sizeof(float));
	}

	std::string to_npy_str() const {
		std::string str = "np.matrix([";
		size_t index = 0;
		for (size_t row = 0; row < rows; row++) {
			str += "[";
			for (size_t col = 0; col < cols; col++) {
				str += std::to_string(values[index]) + ", ";
				index++;
			}
			str += "], ";
		}
		return str + "])";
	}

	void add_row() {
		float* values_new = new float[rows * cols + cols];
		std::memset(values_new, 0, (rows * cols + cols ) * sizeof(float));
		std::memcpy(values_new, values, rows * cols * sizeof(float));
		delete[] values;
		values = values_new;
		rows++;
	}

	void add_col() {
		float* values_new = new float[rows * cols + rows];
		std::memset(values_new, 0, (rows * cols + rows) * sizeof(float));
		for (size_t row = 0; row < rows; row++) {
			std::memcpy(values_new + row * (cols+1), values + row * cols, cols * sizeof(float));
		}
		delete[] values;
		values = values_new;
		cols++;
	}

	float& value(size_t row, size_t col) {
		return values[row * cols + col];
	}

	size_t rows = 0;
	size_t cols = 0;
	float* values = nullptr;
};

//Inlined Vector and Matrix functions

static inline void Vector_add_vec(Vector& u, const Vector& v) {
	assert(u.rows == v.rows);
	for (size_t row = 0; row < u.rows; row++) {
		u.values[row] += v.values[row];
	}
}

static inline void Vector_add_s_mul(Vector& u, float s, const Vector& v) {
	assert(u.rows == v.rows);
	for (size_t row = 0; row < u.rows; row++) {
		u.values[row] += s * v.values[row];
	}
}

static inline void Matrix_add_s_mul(Matrix& u, float s, const Matrix& v) {
	assert(u.rows == v.rows && u.cols == v.cols);
	for (size_t i = 0; i < u.rows * u.cols; i++) {
		u.values[i] += s * v.values[i];
	}
}

static inline void Matrix_add_out_prod(Matrix& u, const Vector& v, const Vector& w) {
	assert(u.rows == v.rows && u.cols == w.rows);
	size_t index = 0;
	for (size_t row = 0; row < u.rows; row++) {
		float v_ = v.values[row];
		for (size_t col = 0; col < u.cols; col++) {
			u.values[index] += v_ * w.values[col];
			index++;
		}
	}
}

static inline void Vector_set_backprop_delta(Vector& d, const Matrix& W, const Vector& d_, const Vector& a) {
	assert(W.rows == d_.rows);
	assert(W.cols == a.rows);
	assert(W.cols == d.rows);
	for (size_t i = 0; i < d.rows; i++) {
		for (size_t j = 0; j < W.rows; j++) {
			d.values[i] += W.values[j * W.cols + i] * d_.values[j];
		}
		d.values[i] *= a.values[i];
	}
}

static inline void Vector_randomize(Vector& u, Random& random) {
	for (size_t row = 0; row < u.rows; row++)
		u.values[row] = random.next_ufloat();
}

static inline void Matrix_randomize(Matrix& u, Random& random) {
	for (size_t i = 0; i < u.rows * u.cols; i++)
		u.values[i] = random.next_ufloat();
}


class TrainLog 
{
public:
	TrainLog(const std::string& filename, size_t max_epochs) :
		m_filename(filename)
	{
		m_epochs = max_epochs;
		m_costs_list.reserve(max_epochs);
	}

	void append(float cost) {
		m_costs_list.push_back(cost);
	}

	void save() const {
		std::ofstream file(m_filename + "_cost_vs_epochs");
		file << "import matplotlib.pyplot as plt\n";
		file << "import numpy as np\n";
		file << "x = np.arange(" << m_epochs << ")\n";
		file << "y = [";
		for (float y : m_costs_list) file << y << ", ";
		file << "]\n";
		file << "plt.plot(x,y,color='Black')\n";
		file << "plt.xlabel('Epochs')\n";
		file << "plt.ylabel('Cost')\n";
		file << "plt.title('Cost vs. Epochs')\n";
		file << "plt.show()";
	}

private:
	std::string m_filename;
	size_t m_epochs;
	std::vector<float> m_costs_list;
};

class NeuralNetwork
{
public:
	/*
	The weights, biases and activation functions can be saved to file in several formats
	BIN -> saved consecutively as raw bytes in a .bin file
	TXT -> saved in human readable txt format with tab separated values
	PY -> saved as a .py file with the layer information, weights, biases and activation functions saved as Python lists of Numpy objects
	*/
	enum class FileFormat {BIN = 0, TXT, PY};
	/*
	There are 10 activation functions available. New ones can be added in the Layer::set_activations() function
	*/
	enum class ActivationFunction {ID = 0, SIGMOID, TANH, RELU, ARCTAN, PRELU, ELU, SOFT_PLUS, BINARY_STEP, SWISH};
	/*
	There are 4 cost functions available. New ones can be added in the NeuralNetwork::cost() function
	*/
	enum class CostFunction {QUADRATIC = 0, CROSS_ENTROPY, EXPONENTIAL, HELINGER};
	
	/*
	TrainingInfo is the struct used to pass in all the training parameters to the NeuralNetwork::train() function
	
	inputs             -> Pointer to the vector of input vectors.
						  Runtime error thrown if size is zero or if there is a size mismatch with expected_outputs or if null
	
	expected_outputs   -> Pointer to the vector of expected outputs.
	                      Runtime error thrown if size is zero or if there is a size mismatch with expected_outputs or if null
	
	epochs             -> Number of epochs (training cycles) of training
	
	epochs_until_save  -> Period of epochs in between each save operation.
	                      You may want to reduce unwanted IO operations
	
	batch_size         -> Size of each batch that the training data will be split into.
	                      NeuralNetwork::train() ensures that batch_size is a divisor of inputs.size().
						  Runtime error thrown if set to zero.
	
	min_cost_threshold -> Minimum cost/loss that when achieved, training will stop
	
	learning_rate      -> A hyperparameter (l) that controls the step size in w -> w + l*dw, b + l*db
	                      This parameter is divided by the batch size before training starts.
						  Runtime error is thrown if learning rate is non-positive
	
	save_log           -> If True, the function of cost/loss vs. epochs is saved to a .py for display using MatPlotLib
	                      Set to False by default

	T_start            -> Starting temperature parameter for the simulated annealing process
						  Set to 2.0f by default

	T_decay            -> The decay parameter used to reduce the temperature T_(t+1) = T_(t) * T_decay
	                      Runtime error thrown if T_decay >= 1.0f
						  Set to 0.99f by default

	T_period           -> The number of steps between each temperature update
						  Set to 1 by default

	T_schedule         -> A function of the form: float T(uint32 step) that provides a schedule for which
						  value of T should be used given the current step
						  No checks are done on the output produced
						  No exceptions should be thrown (however no checks are done on this)
						  T(t) should be decreasing (however no checks are done on this)
						  T(t) should asymptotically approach 0.0f (however no checks are done on this)
						  If T_schedule == NULL then T_decay is used 
						  Set to NULL by default

	max_steps          -> The maximum number of steps to be run for the simulated annealing process
						  Set to 1 by default

	step_size          -> Parameter to modulate the size of the step during the simulated annealing process
						  W(t+1) = W(t) + step_size * dW(t), W are the weights, dW is a random step
						  B(t+1) = B(t) + step_size * dB(t), B are the biases,  dB is a random step
						  Set to 0.001f by default

	*/
	struct TrainingInfo {
		//General
		std::vector<Vector>* inputs = nullptr;            //Pointer to the vector of input vectors
		std::vector<Vector>* expected_outputs = nullptr;  //Pointer to the vector of expected outputs
		float min_cost_threshold = 0.0f;                  //Minimum cost/loss reached before termination. Set to 0.0f by default
		bool save_log = false;                            //Save cost vs. epochs function to Py file. Set to false by default

		//Backpropagation parameters
		uint32_t epochs = 0;                              //Number of epochs (training cycles) of training. Set to 0 by default
		uint32_t epochs_until_save = 1;                   //Period of epochs in between each save operation. Set to 1 by default
		size_t batch_size = 1;                            //Size of each batch that the training data will be split into. Set to 1 by default
		float learning_rate = 0.001f;                     //Step size for gradient descent. Set to 0.001f by default
		
	
		//Simulated Annealing parameters
		float T_start = 2.0f;                             //Starting temperature parameter. Set to 2.0f by default
		float T_decay = 0.99f;                            //Decay parameter used to reduce the temperature T_(t+1) = T_(t) * T_decay. Set to 0.99f by default
		uint32_t T_period = 1;                            //Number of steps between each temperature update. Set to 1 by default
		uint32_t max_steps = 1;                           //Maximum number of steps to be run. Set to 1 by default
		float step_size = 0.001f;                         //Step size for random walk. Set to 0.001f by default
		std::function<float(uint32_t)> T_schedule = nullptr; //temperature schedule function. Set to NULL by default
	
		//Adam optimization
		//float step_size = 0.001f;                         (already included above)
		float beta_1 = 0.001f;							  //exponential decay rate
		float beta_2 = 0.001f;                            //exponential decay rate              
	};

	class Layer
	{
	public:
		struct LayerInfo {
			size_t layer_size = 0;
			ActivationFunction activation_function = ActivationFunction::SIGMOID;
			float activation_function_param = 1.0f;
		};

	public:
		size_t rows = 0;
		Vector activations;
		Vector activations_derivatives;
		Vector biases;
		Matrix weights;
		Matrix cost_grad_weights_sum;
		Vector cost_grad_biases_sum;
		Vector delta;
		ActivationFunction activation_function = ActivationFunction::SIGMOID;
		float activation_function_param = 1.0f;
		Layer* next = nullptr;
		Layer* prev = nullptr;

		Layer() {}

		Layer(size_t rows)
			:rows(rows), activations(rows)
		{
		}

		Layer(size_t rows, size_t cols) :
			rows(rows),
			activations(rows),
			activations_derivatives(rows),
			biases(rows),
			weights(rows, cols),
			cost_grad_weights_sum(rows, cols),
			cost_grad_biases_sum(rows),
			delta(rows)
		{
		}

		Layer(size_t rows, size_t cols, Random& random, ActivationFunction activation_function = ActivationFunction::SIGMOID, float activation_function_param = 1.0f)
			:rows(rows),
			activations(rows),
			activations_derivatives(rows),
			biases(rows,random),
			weights(rows, cols, random),
			cost_grad_weights_sum(rows, cols),
			cost_grad_biases_sum(rows),
			delta(rows),
			activation_function(activation_function),
			activation_function_param(activation_function_param)
		{
		}

		void set_activations() {
			assert(weights.cols == prev->activations.rows);
			assert(weights.rows == activations.rows);
			Vector_set_vec(activations,biases);
			switch (activation_function)
			{
			case ActivationFunction::ID:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					for (size_t col = 0; col < weights.cols; col++) {
						activations.values[row] += weights.values[index] * prev->activations.values[col];
					}
					activations_derivatives.values[row] = 1.0f;
				}
				return;
			}
			case ActivationFunction::SIGMOID:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					float w_sum = 0.0f;
					for (size_t col = 0; col < weights.cols; col++) {
						w_sum += weights.values[index] * prev->activations.values[col];
						index++;
					}
					//sigmoid
					float s = 1.0f / (1.0f + expf(-w_sum));
					activations.values[row] = s;
					activations_derivatives.values[row] = s * (1.0f - s);
				}
				return;
			}
			case ActivationFunction::TANH:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					for (size_t col = 0; col < weights.cols; col++) {
						activations.values[row] += weights.values[index] * prev->activations.values[col];
					}
					activations.values[row] = tanhf(activations.values[row]);
					activations_derivatives.values[row] = 1.0f - activations.values[row] * activations.values[row];
				}
				return;
			}
			case ActivationFunction::RELU:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					for (size_t col = 0; col < weights.cols; col++) {
						activations.values[row] += weights.values[index] * prev->activations.values[col];
						index++;
					}
					float weighted_sum = activations.values[row];
					activations.values[row] = (weighted_sum < 0) ? 0.0f : weighted_sum;
					activations_derivatives.values[row] = (weighted_sum < 0) ? 0.0f : 1.0f;
				}
				return;
			}
			case ActivationFunction::ARCTAN:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					for (size_t col = 0; col < weights.cols; col++) {
						activations.values[row] += weights.values[index] * prev->activations.values[col];
						index++;
					}
					float weighted_sum = activations.values[row];
					activations.values[row] = atanf(weighted_sum);
					activations_derivatives.values[row] = 1.0f / (1.0f + weighted_sum * weighted_sum);
				}
				return;
			}
			case ActivationFunction::PRELU:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					for (size_t col = 0; col < weights.cols; col++) {
						activations.values[row] += weights.values[index] * prev->activations.values[col];
						index++;
					}
					float weighted_sum = activations.values[row];
					activations.values[row] = (weighted_sum < 0) ? activation_function_param * weighted_sum : weighted_sum;
					activations_derivatives.values[row] = (weighted_sum < 0) ? activation_function_param : 1.0f;
				}
				return;
			}
			case ActivationFunction::ELU:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					for (size_t col = 0; col < weights.cols; col++) {
						activations.values[row] += weights.values[index] * prev->activations.values[col];
						index++;
					}
					float weighted_sum = activations.values[row];
					activations.values[row] = (weighted_sum < 0) ? activation_function_param * expf(weighted_sum) - activation_function_param : weighted_sum;
					activations_derivatives.values[row] = (weighted_sum < 0) ? activations.values[row] + activation_function_param : 1.0f;
				}
				return;
			}
			case ActivationFunction::SOFT_PLUS:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					for (size_t col = 0; col < weights.cols; col++) {
						activations.values[row] += weights.values[index] * prev->activations.values[col];
						index++;
					}
					float weighted_sum = activations.values[row];
					activations.values[row] = logf(1.0f + expf(weighted_sum));
					activations_derivatives.values[row] = 1.0f / (1.0f + expf(-weighted_sum));
				}
				return;
			}
			case ActivationFunction::BINARY_STEP:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					for (size_t col = 0; col < weights.cols; col++) {
						activations.values[row] += weights.values[index] * prev->activations.values[col];
						index++;
					}
					float weighted_sum = activations.values[row];
					activations.values[row] = (weighted_sum < 0) ? 0.0f : 1.0f;
					activations_derivatives.values[row] = 0.0f;
				}
				return;
			}
			case ActivationFunction::SWISH:
			{
				size_t index = 0;
				for (size_t row = 0; row < weights.rows; row++) {
					for (size_t col = 0; col < weights.cols; col++) {
						activations.values[row] += weights.values[index] * prev->activations.values[col];
						index++;
					}
					float weighted_sum = activations.values[row];
					activations.values[row] = weighted_sum / (1.0f + expf(-weighted_sum));
					float a = expf(weighted_sum);
					float b = 1.0f / ((1.0f + a) * (1.0f + a));
					activations_derivatives.values[row] = a * (weighted_sum + a + 1.0f) * b;
				}
				return;
			}
			default:
				return;
			}
		}

		void add_node() {
			activations.add_row();
			activations_derivatives.add_row();
			biases.add_row();
			weights.add_row();
			next->weights.add_col();
			cost_grad_biases_sum.add_row();
			cost_grad_weights_sum.add_row();
			next->cost_grad_weights_sum.add_col();
			delta.add_row();
		}
	};

public:
	NeuralNetwork(){}
	NeuralNetwork(const std::vector<Layer::LayerInfo>& layer_info_list, const std::string& filename);
	NeuralNetwork(const std::string& filename, FileFormat format);
	~NeuralNetwork();

	Vector feedforward(const Vector& inputs);
	void save_to_file(FileFormat format) const;
	void train_backprop(TrainingInfo& training_info);
	void train_simulated_annealing(TrainingInfo& training_info);
	void train_adam(TrainingInfo& training_info);

	Layer* input_layer() const {
		return m_input_layer;
	}

	Layer* output_layer() const {
		return m_output_layer;
	}

	std::vector<size_t> layer_sizes() {
		return m_layer_sizes;
	}

	size_t num_layers() const {
		return m_num_layers;
	}

private:
	float cost(const Vector& outputs, const Vector& expected_outputs, Vector& cost_derivative);
	float cost(const Vector& outputs, const Vector& expected_outputs);
	void feedforward_no_ret(const Vector& inputs);

private:
	size_t m_num_layers = 0;
	std::vector<size_t> m_layer_sizes;
	std::string m_filename;
	Layer* m_input_layer = nullptr;
	Layer* m_output_layer = nullptr;
	CostFunction m_cost_function = CostFunction::QUADRATIC;
	float m_cost_function_param = 1.0f;
	float m_fitness = -FLT_MAX;
	Random m_random;
};


//	Main methods of the Neural Network

NeuralNetwork::NeuralNetwork(const std::vector<Layer::LayerInfo>& layer_info_list, const std::string& filename) :
	m_num_layers(layer_info_list.size()),
			m_filename(filename)
{
	m_layer_sizes.resize(m_num_layers);
	m_layer_sizes[0] = layer_info_list[0].layer_size;
	m_input_layer = new Layer(layer_info_list[0].layer_size);
	Layer* curr = m_input_layer;
	for (size_t layer_index = 1; layer_index < layer_info_list.size(); layer_index++) {
		size_t layer_size = layer_info_list[layer_index].layer_size;
		m_layer_sizes[layer_index] = layer_size;
		size_t prev_layer_size = layer_info_list[layer_index-1].layer_size;
		ActivationFunction activation_function = layer_info_list[layer_index].activation_function;
		float activation_function_param = layer_info_list[layer_index].activation_function_param;
		curr->next = new Layer(layer_size, prev_layer_size, m_random, activation_function, activation_function_param);
		curr->next->prev = curr;
		curr = curr->next;
	}
	m_output_layer = curr;
}

NeuralNetwork::NeuralNetwork(const std::string& filename, FileFormat format = FileFormat::BIN)
		:m_filename(filename)
	{
		switch (format)
		{
		case FileFormat::BIN:
		{
			std::ifstream file(filename + ".bin", std::ios::binary);
			file.read((char*)(&m_num_layers), sizeof(size_t));
			m_layer_sizes.resize(m_num_layers);
			file.read((char*)m_layer_sizes.data(), m_num_layers * sizeof(size_t));		
			m_input_layer = new Layer(m_layer_sizes[0]);
			Layer* curr = m_input_layer;
			for (size_t i = 1; i < m_layer_sizes.size(); i++) {
				curr->next = new Layer(m_layer_sizes[i], m_layer_sizes[i - 1]);
				curr->next->prev = curr;
				file.read((char*)curr->next->weights.values, m_layer_sizes[i] * m_layer_sizes[i - 1] * sizeof(float));
				file.read((char*)curr->next->biases.values, m_layer_sizes[i] * sizeof(float));
				file.read((char*)&curr->activation_function, sizeof(int));
				file.read((char*)&curr->activation_function_param, sizeof(float));
				curr = curr->next;
			}
			m_output_layer = curr;
			break;
		}
		case FileFormat::TXT:
		{
			std::ifstream file(filename + ".txt");
			file >> m_num_layers;
			m_layer_sizes.resize(m_num_layers);
			for (size_t i = 0; i < m_num_layers; i++)
				file >> m_layer_sizes[i];
			m_input_layer = new Layer(m_layer_sizes[0]);
			Layer* curr = m_input_layer;
			for (size_t layer_index = 1; layer_index < m_layer_sizes.size(); layer_index++) {
				curr->next = new Layer(m_layer_sizes[layer_index], m_layer_sizes[layer_index - 1]);
				for (size_t j = 0; j < m_layer_sizes[layer_index] * m_layer_sizes[layer_index - 1]; j++)
					file >> curr->next->weights.values[j];
				for (size_t j = 0; j < m_layer_sizes[layer_index]; j++)
					file >> curr->next->biases.values[j];
				int activation_func = 0;
				file >> activation_func;
				curr->activation_function = (ActivationFunction)activation_func;
				file >> curr->activation_function_param;
				curr = curr->next;
			}
			m_output_layer = curr;
			int cost_func = 0;
			file >> cost_func;
			m_cost_function = (CostFunction)cost_func;
			file >> m_cost_function_param;
			break;
		}		
		default:
			break;
		}
	}
 
NeuralNetwork::~NeuralNetwork(){
	Layer* curr = m_input_layer;
	while (curr != nullptr) {
		Layer* next = curr->next;
		delete curr;
		curr = next;
	}
	m_input_layer = nullptr;
	m_output_layer = nullptr;
}
 
Vector NeuralNetwork::feedforward(const Vector& inputs){
	assert(inputs.rows == m_input_layer->rows);
	Layer* curr = m_input_layer;
	Vector_set_vec(curr->activations,inputs);
	curr = curr->next;
	while (curr != nullptr) {
		curr->set_activations();
		curr = curr->next;
	}
	return m_output_layer->activations;
}

void NeuralNetwork::feedforward_no_ret(const Vector& inputs) {
	Layer* curr = m_input_layer;
	Vector_set_vec(curr->activations, inputs);
	curr = curr->next;
	while (curr != nullptr) {
		curr->set_activations();
		curr = curr->next;
	}
}

void NeuralNetwork::save_to_file(FileFormat format = FileFormat::BIN) const {
	switch (format)
	{
	case FileFormat::BIN: {
		//Save parameters as contiguous byte array
		//Number of layers; Layer sizes; Weights, Biases and Activation function for each layer
		std::ofstream file(m_filename + ".bin", std::ios::binary);
		file.write((char*)&m_num_layers, sizeof(size_t));
		file.write((char*)m_layer_sizes.data(), m_num_layers * sizeof(size_t));
		Layer* curr = m_input_layer->next;
		while (curr != nullptr) {
			file.write((char*)curr->weights.values, curr->weights.rows * curr->weights.cols * sizeof(float));
			file.write((char*)curr->biases.values, curr->biases.rows * sizeof(float));
			file.write((char*)&curr->activation_function, sizeof(int));
			file.write((char*)&curr->activation_function_param, sizeof(float));
			curr = curr->next;
		}

		file.close();
		return;
	}
	case FileFormat::TXT:
	{
		//Save parameters as done with BIN but as chars in .txt file for human  readability
		//Number of layers; Layer sizes; Weights, Biases and Activation function for each layer
		std::ofstream file(m_filename + ".txt");
		file << m_num_layers << '\n';
		for (size_t layer_size : m_layer_sizes) file << layer_size << ' ';
		file << '\n';
		Layer* curr = m_input_layer->next;
		while (curr != nullptr) {
			for (size_t i = 0; i < curr->weights.rows * curr->weights.cols; i++)
				file << curr->weights.values[i] << '\t';
			file << '\n';
			for (size_t i = 0; i < curr->biases.rows; i++)
				file << curr->biases.values[i] << '\t';
			file << '\n';
			file << (int)curr->activation_function << ' ';
			file << (float)curr->activation_function_param << '\n';
			curr = curr->next;
		}
		file << (int)m_cost_function << '\n';
		file << m_cost_function_param;
		file.close();
		return;
	}
	case FileFormat::PY:
	{
		//Save as python file were the arrays are stored as PY lists and vectors/matrices are stored as NUMPY arrays/matrices
		//This can not be used in the NeuralNetwork constructor (yet)
		std::ofstream file(m_filename + ".py");
		file << "import numpy as np\n";
		file << "num_layers = " << m_num_layers << '\n';
		file << "layer_sizes = [";
		for (size_t layer_size : m_layer_sizes) file << layer_size << ", ";
		file << "]\n";

		file << "weights = [";
		Layer* curr = m_input_layer->next;
		while (curr != nullptr) {
			file << curr->weights.to_npy_str() << ", ";
			curr = curr->next;
		}
		file << "]\n";

		file << "biases = [";
		curr = m_input_layer->next;
		while (curr != nullptr) {
			file << curr->biases.to_npy_str() << ", ";
			curr = curr->next;
		}
		file << "]\n";

		const std::string activation_functions_str[10] = {"ID", "SIGMOID","TANH","RELU","ARCTAN","PRELU","ELU","SOFT_PLUS","BINARY_STEP","SWISH"};

		file << "activation_functions = [";
		curr = m_input_layer->next;
		while (curr != nullptr) {
			file << "('" << activation_functions_str[(int)curr->activation_function] << "', " << curr->activation_function_param << "), ";
			curr = curr->next;
		}
		file << "]\n";

		const std::string cost_functions_str[4] = { "QUADRATIC", "CROSS_ENTROPY", "EXPONENTIAL","HELINGER"};
		file << "cost_function = '" << cost_functions_str[(int)m_cost_function] << "'\n";
		file << "cost_function_param = " << m_cost_function_param;
		file.close();

		break;
	}
	default:
		return;
	}
}

float NeuralNetwork::cost(const Vector& outputs, const Vector& expected_outputs, Vector& cost_derivative) {
	float cost = 0.0f;
	switch (m_cost_function)
	{
	case CostFunction::QUADRATIC:
	{
		for (size_t row = 0; row < outputs.rows; row++) {
			float diff = outputs.values[row] - expected_outputs.values[row];
			cost_derivative.values[row] = diff;
			cost += diff * diff;
		}
		return 0.5f * cost;
	}
	case CostFunction::CROSS_ENTROPY:
	{
		for (size_t row = 0; row < outputs.rows; row++) {
			cost -= expected_outputs.values[row] * logf(outputs.values[row] +
				(1.0f - expected_outputs.values[row]) * logf(1.0f - outputs.values[row]));
			cost_derivative.values[row] = (outputs.values[row] - expected_outputs.values[row]) / (outputs.values[row] - outputs.values[row]* outputs.values[row]);
		}
		return cost;
	}
	case CostFunction::EXPONENTIAL:
	{
		for (size_t row = 0; row < outputs.rows; row++) {
			float diff = outputs.values[row] - expected_outputs.values[row];
			cost_derivative.values[row] = diff;
			cost += diff * diff;
		}
		for (size_t row = 0; row < outputs.rows; row++) {
			cost_derivative.values[row] *= cost;
		}
	
		return cost * expf(cost / m_cost_function_param);
	}
	case CostFunction::HELINGER:
	{
		for (size_t row = 0; row < outputs.rows; row++) {
			float diff = sqrtf(outputs.values[row]) - sqrtf(expected_outputs.values[row]);
			cost += diff * diff;
			cost_derivative.values[row] = 1.0f - sqrt(expected_outputs.values[row] / outputs.values[row]);
		}
		return 0.5f * cost;
	}
	}
	return cost;
}

float NeuralNetwork::cost(const Vector& outputs, const Vector& expected_outputs) {
	float cost = 0.0f;
	switch (m_cost_function)
	{
	case CostFunction::QUADRATIC:
	{
		for (size_t row = 0; row < outputs.rows; row++) {
			float diff = outputs.values[row] - expected_outputs.values[row];
			cost += diff * diff;
		}
		return 0.5f * cost;
	}
	case CostFunction::CROSS_ENTROPY:
	{
		for (size_t row = 0; row < outputs.rows; row++) {
			cost -= expected_outputs.values[row] * logf(outputs.values[row] +
				(1.0f - expected_outputs.values[row]) * logf(1.0f - outputs.values[row]));
		}
		return cost;
	}
	case CostFunction::EXPONENTIAL:
	{
		for (size_t row = 0; row < outputs.rows; row++) {
			float diff = outputs.values[row] - expected_outputs.values[row];
			cost += diff * diff;
		}

		return cost * expf(cost / m_cost_function_param);
	}
	case CostFunction::HELINGER:
	{
		for (size_t row = 0; row < outputs.rows; row++) {
			float diff = sqrtf(outputs.values[row]) - sqrtf(expected_outputs.values[row]);
			cost += diff * diff;
		}
		return 0.5f * cost;
	}
	}
	return cost;
}

void NeuralNetwork::train_backprop(TrainingInfo& training_info) {
	if (training_info.inputs == nullptr || training_info.expected_outputs == nullptr)
		throw std::runtime_error("Inputs and/or Expected Outputs null");
	if (training_info.inputs->size() == 0 || training_info.expected_outputs->size() == 0)
		throw std::runtime_error("Inputs or Expected Outputs has size 0");
	if (training_info.inputs->size() != training_info.expected_outputs->size())
		throw std::runtime_error("Inputs and Expected Outputs size mismatch");
	if (training_info.batch_size == 0)
		throw std::runtime_error("Zero batch size");
	if (training_info.learning_rate <= 0.0f)
		throw std::runtime_error("Learning rate is non-positive");

	while (training_info.inputs->size() % training_info.batch_size != 0)
		training_info.batch_size++;
	training_info.learning_rate /= (float)training_info.batch_size;

	size_t num_batches = training_info.inputs->size() / training_info.batch_size;
	TrainLog train_log(m_filename, training_info.epochs);

	float duration_between_prints = 0.0f;
	uint32_t current_epoch = 0u;
	float   current_cost = FLT_MAX;
	Vector  cost_derivative(training_info.expected_outputs->at(0).rows);

	//Shuffle the inputs by generating a random indexing set
	std::vector<size_t> index_set(training_info.inputs->size());
	for (size_t i = 0; i < index_set.size(); i++) index_set[i] = i;

	std::vector<Matrix> weights_updates;
	std::vector<Vector> biases_updates;

	weights_updates.reserve(m_num_layers - 1);
	biases_updates.reserve(m_num_layers - 1);
	for (size_t i = 1; i < m_num_layers; i++) {
		weights_updates.emplace_back(m_layer_sizes[i], m_layer_sizes[i - 1]);
		biases_updates.emplace_back(m_layer_sizes[i]);
	}

	while (current_epoch < training_info.epochs && current_cost > training_info.min_cost_threshold) {
		float duration = 0.0f;
		{
			Timer timer(&duration);

			current_cost = 0.0f;

			//Shuffle the inputs by generating a random indexing set
			m_random.shuffle<size_t>(index_set);

			size_t count = 0;
			for (size_t index = 0; index < training_info.inputs->size(); index++) {
				feedforward_no_ret(training_info.inputs->at(index));
				current_cost += cost(m_output_layer->activations, training_info.expected_outputs->at(index), cost_derivative);

				//Backpropagation
				size_t L = weights_updates.size() - 1;
				Layer* curr = m_output_layer;
				for (int row = 0; row < curr->delta.rows; row++) {
					curr->delta.values[row] = cost_derivative.values[row] * curr->activations_derivatives.values[row];
					biases_updates[L].values[row] += curr->delta.values[row];
				}
				Matrix_add_out_prod(weights_updates[L], curr->delta, curr->prev->activations);
				curr = curr->prev;
				L--;
				while (curr->prev != nullptr) {
					Vector_set_backprop_delta(curr->delta, curr->next->weights, curr->next->delta, curr->activations_derivatives);
					for (int row = 0; row < curr->delta.rows; row++) {
						biases_updates[L].values[row] += curr->delta.values[row];
					}
					Matrix_add_out_prod(weights_updates[L], curr->delta, curr->prev->activations);
					curr = curr->prev;
					L--;
				}

				count++;
				if (count % training_info.batch_size==0) {
					//Apply gradients
					Layer* curr = m_input_layer->next;
					size_t L = 0;
					while (curr != nullptr) {
						for (int row = 0; row < curr->biases.rows; row++) {
							curr->biaes.values[row] -= training_info.learning_rate * biases_updates[L].values[row];
						}
						Matrix_add_s_mul(curr->weights, -training_info.learning_rate, weights_updates[L]);
						biases_updates[L].set_zero();
						weights_updates[L].set_zero();
						curr = curr->next;
						L++;
					}
				}

			}

		}
		duration_between_prints += duration;

		if (current_epoch % training_info.epochs_until_save == 0) {
			std::cout
				<< "EPOCH:" << current_epoch << "\t"
				<< "DURATION:" << duration_between_prints << "ms\t"
				<< "COST:" << current_cost << "\n";

			duration_between_prints = 0.0f;

			if (training_info.save_log)
				train_log.append(current_cost);

			save_to_file();
		}
		current_epoch++;

	}
	
	if (training_info.save_log)
		train_log.save();
}

void NeuralNetwork::train_simulated_annealing(TrainingInfo& training_info) {
	if (training_info.inputs == nullptr || training_info.expected_outputs == nullptr)
		throw std::runtime_error("Inputs and/or Expected Outputs null");
	if (training_info.inputs->size() == 0 || training_info.expected_outputs->size() == 0)
		throw std::runtime_error("Inputs or Expected Outputs has size 0");
	if (training_info.inputs->size() != training_info.expected_outputs->size())
		throw std::runtime_error("Inputs and Expected Outputs size mismatch");
	if (training_info.step_size < 0)
		throw std::runtime_error("Step size negative");
	if (training_info.T_decay >= 1.0f)
		throw std::runtime_error("T_decay >= 1.0f");

	float T = training_info.T_start;

	std::vector<Matrix> weights_delta;
	std::vector<Vector> biases_delta;

	weights_delta.reserve(m_num_layers - 1);
	biases_delta.reserve(m_num_layers - 1);
	for (size_t i = 1; i < m_num_layers; i++) {
		weights_delta.emplace_back(m_layer_sizes[i], m_layer_sizes[i - 1],m_random);
		biases_delta.emplace_back(m_layer_sizes[i],m_random);
	}

	float cost_prev = 0.0f;
	for (size_t i = 0; i < training_info.inputs->size(); i++) {
		cost_prev += cost(feedforward(training_info.inputs->at(i)), training_info.expected_outputs->at(i));
	}

	for (size_t step = 0; step < training_info.max_steps; step++) {

		Layer* curr = m_input_layer->next;
		size_t L = 0;
		while (curr != nullptr) {
			Matrix_randomize(weights_delta[L],m_random);
			Vector_randomize(biases_delta[L], m_random);
			Matrix_add_s_mul(curr->weights, training_info.step_size, weights_delta[L]);
			Vector_add_s_mul(curr->biases, training_info.step_size, biases_delta[L]);
			curr = curr->next;
			L++;
		}

		float cost_curr = 0.0f;
		for (size_t i = 0; i < training_info.inputs->size(); i++) {
			cost_curr += cost(feedforward(training_info.inputs->at(i)), training_info.expected_outputs->at(i));
		}

		float cost_delta = cost_curr - cost_prev;
		float u = m_random.next_float();

		if (cost_delta < 0 || u < expf(-cost_delta / T)) {
			//Keep
			cost_prev = cost_curr;
		}
		else {
			//Reject
			Layer* curr = m_input_layer->next;
			size_t L = 0;
			while (curr != nullptr) {
				Matrix_add_s_mul(curr->weights, -training_info.step_size, weights_delta[L]);
				Vector_add_s_mul(curr->biases, -training_info.step_size, biases_delta[L]);
				curr = curr->next;
				L++;
			}
		}

		if (step % training_info.T_period == 0 && step > 0) {
			if (training_info.T_schedule != nullptr) {
				T = training_info.T_schedule((uint32_t)step);
			}
			else {
				T *= training_info.T_decay;
			}	
			std::cout << "T=" << T << "\t COST=" <<  cost_curr << "\n";
		}

	}
}

void NeuralNetwork::train_adam(TrainingInfo& training_info) {
	if (training_info.inputs == nullptr || training_info.expected_outputs == nullptr)
		throw std::runtime_error("Inputs and/or Expected Outputs null");
	if (training_info.inputs->size() == 0 || training_info.expected_outputs->size() == 0)
		throw std::runtime_error("Inputs or Expected Outputs has size 0");
	if (training_info.inputs->size() != training_info.expected_outputs->size())
		throw std::runtime_error("Inputs and Expected Outputs size mismatch");
	if (training_info.step_size < 0)
		throw std::runtime_error("Step size negative");
	if (training_info.beta_1 >= 1.0f || training_info.beta_1 < 0.0f || 
		training_info.beta_2 >= 1.0f || training_info.beta_2 < 0.0f)
		throw std::runtime_error("Beta decay factors out of bounds");

	std::vector<Matrix> weights_grad;
	std::vector<Vector> biases_grad;
	weights_grad.reserve(m_num_layers - 1);
	biases_grad.reserve(m_num_layers - 1);
	for (size_t i = 1; i < m_num_layers; i++) {
		weights_grad.emplace_back(m_layer_sizes[i], m_layer_sizes[i - 1]);
		biases_grad.emplace_back(m_layer_sizes[i]);
	}

	std::vector<Matrix> m_weights;
	std::vector<Vector> m_biases;
	m_weights.reserve(m_num_layers - 1);
	m_biases.reserve(m_num_layers - 1);
	for (size_t i = 1; i < m_num_layers; i++) {
		m_weights.emplace_back(m_layer_sizes[i], m_layer_sizes[i - 1]);
		m_biases.emplace_back(m_layer_sizes[i]);
	}

	std::vector<Matrix> v_weights;
	std::vector<Vector> v_biases;
	v_weights.reserve(m_num_layers - 1);
	v_biases.reserve(m_num_layers - 1);
	for (size_t i = 1; i < m_num_layers; i++) {
		v_weights.emplace_back(m_layer_sizes[i], m_layer_sizes[i - 1]);
		v_biases.emplace_back(m_layer_sizes[i]);
	}

	uint32_t current_step = 0;
	float    current_cost = FLT_MAX;
	Vector   cost_derivative(training_info.expected_outputs->at(0).rows);
	float    b1 = training_info.beta_1;
	float    b2 = training_info.beta_2;

	while (current_step < training_info.max_steps && current_cost > training_info.min_cost_threshold)
	{
		current_cost = 0.0f;

		//Get gradients g
		for (size_t index = 0; index < training_info.inputs->size(); index++) {
			feedforward_no_ret(training_info.inputs->at(index));
			current_cost += cost(m_output_layer->activations, training_info.expected_outputs->at(index), cost_derivative);
			size_t L = weights_grad.size() - 1;
			Layer* curr = m_output_layer;
			Vector_set_to_had_prod(curr->delta, cost_derivative, curr->activations_derivatives);
			Vector_add_vec(biases_grad[L], curr->delta);
			Matrix_add_out_prod(weights_grad[L], curr->delta, curr->prev->activations);
			curr = curr->prev;
			L--;
			while (curr->prev != nullptr) {
				Vector_set_backprop_delta(curr->delta, curr->next->weights, curr->next->delta, curr->activations_derivatives);
				Vector_add_vec(biases_grad[L], curr->delta);
				Matrix_add_out_prod(weights_grad[L], curr->delta, curr->prev->activations);
				curr = curr->prev;
				L--;
			}
		}

		//Update m and v
		float m = 0.0f;
		float v = 0.0f;
		float m_corrected = 0.0f;
		float v_corrected = 0.0f;

		Layer* curr = m_input_layer->next;
		size_t l = 0;
		while (curr != nullptr) {
			size_t index = 0;
			for (size_t row = 0; row < curr->weights.rows; row++) {
				m = b1 * m_biases[l].values[row] + (1.0f - b1) * biases_grad[l].values[row];
				v = b2 * v_biases[l].values[row] + (1.0f - b2) * biases_grad[l].values[row] * biases_grad[l].values[row];
				m_corrected = m / (1.0f - b1);
				v_corrected = v / (1.0f - b2);
				m_biases[l].values[row] = m;
				v_biases[l].values[row] = v;
				curr->biases.values[row] -= training_info.step_size * m_corrected / (0.001f + sqrtf(v_corrected));
				for (size_t col = 0; col < curr->weights.cols; col++) {
					m = b1 * m_weights[l].values[index] + (1.0f - b1) * weights_grad[l].values[index];
					v = b2 * v_weights[l].values[index] + (1.0f - b2) * weights_grad[l].values[index] * weights_grad[l].values[index];
					m_corrected = m / (1.0f - b1);
					v_corrected = v / (1.0f - b2);
					curr->weights.values[index] -= training_info.step_size * m_corrected / (0.001f + sqrtf(v_corrected));
					m_weights[l].values[index] = m;
					v_weights[l].values[index] = v;
					index++;
				}
			}
			biases_grad[l].set_zero();
			weights_grad[l].set_zero();
			curr = curr->next;
			l++;
		}

		std::cout << "Step:" << current_step << ", Cost:" << current_cost << "\n";
		current_step++;
	}
}