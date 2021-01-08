#include <iostream>
#include <ctime>
#include <math.h>
#include "mpi.h"

double Random(double min, double max);
void printMatrix(double** matrix, int size);
void printVector(double* vec, int size);

double Random(double min, double max)
{
    srand(time(0));
    return min + (rand() % static_cast<int>(max - min + 1));
}

void printVector(double* vec, int size) {
    for (int idx = 0; idx < size; idx++) {
           std::cout << vec[idx] << " ";
    }
    std::cout << std::endl;
}

void printMatrix(double** matrix, int size) {
    for (int idx = 0; idx < size-1; idx++) {
        for (int jdx = 0; jdx < size; jdx++) {
            std::cout << matrix[idx][jdx] << " ";
        }
        std::cout << std::endl;
    }
}

double get_euclidean_norm(double** extend_matrix, double *answ_vec, int size) {
    double norm = 0.0;
    for (int row = 0; row < size - 1; row++) {
        double b_i = extend_matrix[row][size - 1];
        double scalar = 0.0;
        for (int col = 0; col < size - 1; col++) { 
            scalar = scalar + extend_matrix[row][col] * answ_vec[col];
        }
        norm = norm + pow(b_i - scalar, 2);
    }
    return sqrt(norm);
}

double** createExtendMatrix(int size, double min, double max) {
    double** array = new double* [size];
    for (int idx = 0; idx < size - 1; idx++) {
        array[idx] = new double[size];
    }
    for (int row = 0; row < size - 1; row++) {
        double col_sum = 0.0;
        for (int column = 0; column < size; column++) {
            double element = Random(min, max);
            col_sum = col_sum + abs(element);
            array[row][column] = element;
        }
        array[row][row] = col_sum + 1.0;
    }
    return array;
}

double* applyYakobi(double** extendmatrix, double* init_conditions, int max_iter_size, int size, double eps, int world_size, int world_rank, MPI_Status& status) 
{    
    int matrixASize = size - 1;
    double d_i = 0.0, b_i = 0.0;
    short int num_iter = 0;
    int part_array = matrixASize / world_size;
    int* counts = new int[world_size]();
    double* answer;
    int proc_lower_limit = 0, proc_upper_limit = part_array - 1;
    if (world_rank != 0) {
        proc_lower_limit = world_rank * part_array;
	    if (world_rank == world_size - 1) {
	        proc_upper_limit = matrixASize - 1;
	    }
	    else {
	        proc_upper_limit = (world_rank + 1) * part_array - 1;
	    }
    }
    std::cout << "rank: " << world_rank << " low_limit: " << proc_lower_limit << " up_limit: " << proc_upper_limit << std::endl;
	
    // смещения для принимающего массива в MPI_Allgatherv 
    int* displacements = new int[world_size]();
    for (int idx = 1; idx < world_size; idx++) {
        displacements[idx] = part_array * idx;
	    //std::cout << "rank: " << world_rank << " disp: " << displacements[idx] << std::endl;
    }

    // размеры массивов, которые будут отправляться от каждого процессора через MPI_Allgatherv
    for (int idx = 0; idx < world_size - 1; idx++) {
        counts[idx] = part_array;
    }
    counts[world_size - 1] = matrixASize - (world_size - 1) * part_array;
    std::cout << "counts: ";
    for (int j = 0; j < world_size; j++) {
	    std::cout << counts[j] << " ";
    }
    std::cout << std::endl;
    while (num_iter <= max_iter_size) {
        answer = new double[counts[world_rank]];
	    /*std::cout << "iter: " << num_iter << " rank: " << world_rank << " init cond: ";
           for (int i = 0; i < matrixASize; i++) {
	        std::cout << init_conditions[i] << " ";
	    }
	    std::cout << std::endl;*/
	    int answ_idx = 0;
	    for (int row = 0; row < matrixASize; row++) {
            if (row < proc_lower_limit || row > proc_upper_limit) { continue; }

	        //std::cout << "rank: " << world_rank << " rowidx: " << row << std::endl;
            b_i = extendmatrix[row][size - 1];
            d_i = (1 / extendmatrix[row][row]);
            double h_i = 0.0;
            for (int col = 0; col < matrixASize; col++) {
                if (row == col) { continue; }
                h_i = h_i - init_conditions[col] * extendmatrix[row][col];
            }
            answer[answ_idx] = d_i * (h_i + b_i);
            answ_idx = answ_idx + 1;
        }
        //std::cout << "result: k = " << num_iter << "| " << "rank: " << world_rank << " part array: " << part_array << std::endl;
       // printVector(answer, counts[world_rank]);
	            
	    MPI_Allgatherv(answer, counts[world_rank], MPI_DOUBLE, init_conditions, counts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
	    MPI_Barrier(MPI_COMM_WORLD);        
	
	    double norm = get_euclidean_norm(extendmatrix, init_conditions, size);
        if (world_rank == 0) {
	        std::cout << "iter: " << num_iter << " norm: " << norm << std::endl;
	    }
	    if (norm < eps) { break; }
        num_iter = num_iter + 1;
        
    }

    delete[] counts;
    delete[] displacements;
    
    if (world_rank == 0) {
	    std::cout << "yakobi end" << std::endl;
    }
    return init_conditions;
}


double* applyZeydel(double** extendmatrix, double* init_conditions, int max_iter_size, int size, double eps, int world_size, int world_rank, MPI_Status& status, int& tag)
{
    int matrixASize = size - 1;
    double d_i = 0.0, b_i = 0.0;
    short int num_iter = 0;
    double* eps_buf = new double[world_size];
    double* norm_buf = new double[1];
    int current_proc = 0;
    bool reverse_way = false;
    while (num_iter < max_iter_size) {
        // 0, 1
        if (current_proc >= world_size - 1) {
            reverse_way = true;
        }
        else if (current_proc == 0) {
            reverse_way = false;
        }

        int target_proc, last_proc;
        if (!reverse_way) {
            // 0
            target_proc = current_proc + 1;
            last_proc = world_rank - 1;
        }
        else {
            target_proc = current_proc - 1;
            last_proc = world_rank + 1;
        }

	    MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == current_proc) {
            for (int row = 0; row < matrixASize; row++) {
                b_i = extendmatrix[row][size - 1];
                d_i = (1 / extendmatrix[row][row]);
                double h_i = 0.0;
                for (int col = 0; col < matrixASize; col++) {
                    if (row == col) { continue; }
                    h_i = h_i - init_conditions[col] * extendmatrix[row][col];
                }
                init_conditions[row] = d_i * (h_i + b_i);
            }
            
            //MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "rank: " << world_rank << " iter: " << num_iter << " send to: " << target_proc << std::endl;
            MPI_Send(init_conditions, matrixASize, MPI_DOUBLE, target_proc, tag, MPI_COMM_WORLD);
        }
        else if (world_rank == target_proc) {
           //MPI_Barrier(MPI_COMM_WORLD
            std::cout << "rank: " << world_rank << " iter: " << num_iter << " receive from: " << last_proc << std::endl;
	        MPI_Recv(init_conditions, matrixASize, MPI_DOUBLE, last_proc, tag, MPI_COMM_WORLD, &status);
	    }

        MPI_Barrier(MPI_COMM_WORLD);

        if (!reverse_way) {
            current_proc = current_proc + 1;
        }
        else {
            current_proc = current_proc - 1;
        }

        norm_buf[0] = get_euclidean_norm(extendmatrix, init_conditions, size);
        MPI_Allgather(norm_buf, 1, MPI_DOUBLE, eps_buf, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == 0) {
            std::cout << "current_proc proc: " << current_proc << " iter: " << num_iter << " norm: " << eps_buf[current_proc] << std::endl;
        }

        if (eps_buf[current_proc] < eps) {
            std::cout << "current_proc end" << " current_proc: " << current_proc << " eps: " << eps_buf[current_proc] << std::endl;
            break;
        }
        num_iter = num_iter + 1;
    } 
    return init_conditions;
}

int main()
{
    int init, tag = 0;

    // Initialize the MPI environment
    init = MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    MPI_Status status;

    if (init != MPI_SUCCESS)
    {
        std::cout << "\nERROR initializing MPI. Exit.\n";
        MPI_Abort(MPI_COMM_WORLD, init);
        return 0;
    }
    // extend matrix size
    const int SIZE = 500;
    const double MIN_MATRIX_VALUE = -50;
    const double MAX_MATRIX_VALUE = 50;
    const int MAX_ITER_SIZE = 1500;
    const double EPS = 1e-4;
    double** array = createExtendMatrix(SIZE, MIN_MATRIX_VALUE, MAX_MATRIX_VALUE);
    double* init_conditions = new double[SIZE - 1];
    double* answer_yakobi = new double[SIZE - 1];
    double* answer_zeydel = new double[SIZE - 1];
    for (int idx = 0; idx < SIZE - 1; idx++) { init_conditions[idx] = 1; }

   // if (world_rank == 0) {
   //    printMatrix(array, SIZE);
   // }
    answer_zeydel = applyZeydel(array, init_conditions, MAX_ITER_SIZE, SIZE, EPS, world_size, world_rank, status, tag);
    // answer_yakobi = applyYakobi(array, init_conditions, MAX_ITER_SIZE, SIZE, EPS, world_size, world_rank, status);
    
    // clear memory
    for (int i = 0; i < SIZE - 1; i++) { delete[] array[i]; }
    delete[] array;
    delete[] init_conditions;
    MPI_Finalize();
}
