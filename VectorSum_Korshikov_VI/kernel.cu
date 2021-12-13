#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h> 
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define SIZE 128						// определяем размерность блока (число потоков)
#define SHMEM_SIZE SIZE * 4				// размер разделяемой памяти

// Задаём функцию для последней итерации, чтобы программа не выполняла лишнюю работу
// volatile нужен, чтобы недопустить кэширование в регистрах (оптимизация)
// В этом случае отпадает необходимость в __syncthreads();
__device__ void warpReduce(volatile int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}

// задаём функцию ядра
// выполняем суммирования элементов массива на графическом процессоре
__global__ void sum_reduction(int* v, int* v_r) {
	// выделяем shared(разделённую) памяти 
	__shared__ int partial_sum[SHMEM_SIZE];
	// расчитываем идентификатор потока
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// загружаем элементы
	// расчитываем масштабируемый индекс, потому что размер массива в два раза превышает количество потоков
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	// Сохраняем первую подсчитанную частичную сумму, не считая элементов
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();
	// цикл начинается с половины блока и делится на две части каждую итерацию
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		// Каждый поток выполняет свою работу в пределах шага
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x < 32) {
		warpReduce(partial_sum, threadIdx.x);
	}
	// В поток 0 мы закидываем всю сумму, которая будет писаться в память
	// Результат вычисление индексируется этим блоком
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

//функция иницализации массива
// заполняем массив единицами, чтобы было удобнее проверять
void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;
	}
}

// функция для вычисления суммы элементов массива на центральном процессоре
int sumVectorCPU(int* v, int n)
{
	int sum = 0;
	for (int i = 0; i < n; i++) {
		sum += v[i];
	}
	return sum;
}

int main()
{
	// ЗАДАЁМ РАЗМЕРНОСТЬ ТЕСТИРУЕМОГО ВЕКТОРА
	int n = 1 << 16;
	size_t bytes = n * sizeof(int);

	// задаём переменную, куда запишем результат вычисленй на центральном процессоре
	int sumCPU = 0;

	// переменные события для работы со временем на графическом процессоре
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// задаём массивы для хранения и обработки данных
	// h_ - работа на центральном процессоре
	// d_ - работа на графическом процессоре
	// постфикс _r - вектор, содержащий результирующие данные 
	int* h_v, * h_v_r;
	int* d_v, * d_v_r;

	// выделяем память под массивы
	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	// инициализируем вектор
	initialize_vector(h_v, n);

	// находим время работы функции, которая вычисляет сумму элементов массива на ЦП
	// high_resolution_clock позволяет точнее определять время
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	sumCPU = sumVectorCPU(h_v, n);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t2 - t1;
	double cpu_time = time_span.count();
	printf("The time: %f milliseconds\n", cpu_time);

	// копируем данные в память ГП
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	// определим размер сетки
	int TB_SIZE = SIZE;
	int GRID_SIZE = n / TB_SIZE / 2;

	// проведение эксперимента на GPU
	cudaEventRecord(start, 0);
	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);
	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %f milliseconds\n", KernelTime);

	// расчитываем ускорение
	double S = cpu_time / KernelTime;
	printf("Acceleration: %f\n", S);

	// копируем данные в оперативную память
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	// проверка результатов
	printf("Accumulated result is %d \n", h_v_r[0]);
	assert(h_v_r[0] == sumCPU);
	printf("COMPLETED SUCCESSFULLY\n");

	// очистка памяти
	cudaFree(d_v);
	cudaFree(d_v_r);
	free(h_v);
	free(h_v_r);
	return 0;
}