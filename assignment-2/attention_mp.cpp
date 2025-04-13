#include <iostream>
#include <vector>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstdlib>
#include <pthread.h>
#include <chrono>

using namespace std;

int total_thread_num = 2;

struct ThreadData {
    int start_row;
    int end_row;
    const vector<vector<int>>* A;
    const vector<vector<int>>* B;
    vector<vector<int>>* result;
};

void* thread_matmul(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int a_cols = data->A->at(0).size();
    int b_cols = data->B->at(0).size();
    for (int i = data->start_row; i < data->end_row; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                data->result->at(i)[j] += data->A->at(i)[k] * data->B->at(k)[j];
    return nullptr;
}

void parallel_matmul(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result) {
    int rows = A.size();
    int rows_per_thread = (rows + total_thread_num - 1) / total_thread_num;

    vector<pthread_t> threads(total_thread_num);
    vector<ThreadData> thread_data(total_thread_num);

    for (int t = 0; t < total_thread_num; ++t) {
        int start = t * rows_per_thread;
        int end = min(start + rows_per_thread, rows);
        thread_data[t] = { start, end, &A, &B, &result };
        pthread_create(&threads[t], nullptr, thread_matmul, &thread_data[t]);
    }

    for (int t = 0; t < total_thread_num; ++t)
        pthread_join(threads[t], nullptr);
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        cerr << "Usage: ./attention_mp <shm_id> <unused_offset> <out_rows> <out_cols> <head_index> <num_heads>" << endl;
        return 1;
    }

    int shm_id = stoi(argv[1]);
    int out_rows = stoi(argv[2]);
    int out_cols = stoi(argv[3]);
    int head_index = stoi(argv[4]);
    int num_heads = stoi(argv[5]);

    int* shm_ptr = (int*)shmat(shm_id, NULL, 0);

    int q_r, q_s, k_r, k_s, v_r, v_s;
    cin >> q_r >> q_s;
    vector<vector<int>> Q(q_r, vector<int>(q_s));
    for (auto& row : Q)
        for (int& x : row) cin >> x;

    cin >> k_r >> k_s;
    vector<vector<int>> K(k_r, vector<int>(k_s));
    for (auto& row : K)
        for (int& x : row) cin >> x;

    cin >> v_r >> v_s;
    vector<vector<int>> V(v_r, vector<int>(v_s));
    for (auto& row : V)
        for (int& x : row) cin >> x;

    auto start_time = chrono::steady_clock::now();

    // K transpose
    vector<vector<int>> K_T(k_s, vector<int>(k_r));
    for (int i = 0; i < k_r; ++i)
        for (int j = 0; j < k_s; ++j)
            K_T[j][i] = K[i][j];

    // Q * K^T
    vector<vector<int>> QK_T(q_r, vector<int>(k_r, 0));
    parallel_matmul(Q, K_T, QK_T);

    // (QK^T) * V
    vector<vector<int>> output(q_r, vector<int>(v_s, 0));
    parallel_matmul(QK_T, V, output);

    auto end_time = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    shm_ptr[head_index] = duration;

    // 변경된 저장 방식: head_index 기반 offset으로 저장
    int* result_ptr = shm_ptr + num_heads;
    for (int i = 0; i < out_rows; ++i)
        for (int j = 0; j < out_cols; ++j)
            result_ptr[head_index * out_rows * out_cols + i * out_cols + j] = output[i][j];

    shmdt(shm_ptr);
    return 0;
}