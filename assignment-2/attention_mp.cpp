#include <iostream>
#include <vector>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstdlib>
#include <pthread.h>
#include <algorithm>

using namespace std;

// 스레드 수 지정
static int total_thread_num = 2;

// 행렬 입력 함수
vector<vector<int>> input_matrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            cin >> matrix[i][j];
        }
    }
    return matrix;
}

struct ThreadData {
    int start_row; // 스레드가 처리할 시작 행
    int end_row; // 스레드가 처리할 끝 행
    const vector<vector<int>>* A;
    const vector<vector<int>>* B;
    vector<vector<int>>* C;
    bool multiplyKT; // 전치를 수행하고 해야하는지 판단하는 변수
};


void* thread_matmul(void* arg) {
    // 스레드에 전달된 데이터 구조체를 가져옴
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);

    int start = data->start_row;
    int end   = data->end_row;

    // A 행렬의 열 개수
    int A_cols = data->A->at(0).size();
    // B 행렬의 열 개수
    int B_cols = data->B->at(0).size();

    if (data->multiplyKT) { // 전치를 하고 행렬 곱셈을 계산
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < static_cast<int>(data->B->size()); j++) {
                int sum = 0;
                for (int x = 0; x < A_cols; x++) { // 전치 후 행렬 곱셈 계산
                    sum += (*data->A)[i][x] * (*data->B)[j][x];
                }
                // 계산 결과 저장
                (*data->C)[i][j] = sum;
            }
        }
    }
    else { // 전치 없이 행렬 곱셈 계산
        for (int i = start; i < end; ++i) {
            for (int x = 0; x < A_cols; x++) {
                int tmp = (*data->A)[i][x]; // A[i][x] 저장
                const auto& bRow = (*data->B)[x]; // B의 x번째 행
                auto& cRow = (*data->C)[i]; // 결과 행렬의 i번째 행
                // 계산 후 결과 저장
                for (int j = 0; j < B_cols; ++j) {
                    cRow[j] += tmp * bRow[j];
                }
            }
        }
    }
    return nullptr;
}

// 행렬 곱셈을 멀티 스레드로 수행하는 함수
void parallel_matmul(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, bool multiplyKT)
{
    int total_rows = A.size(); // 전체 행 개수
    int rows_per_thread = (total_rows + total_thread_num - 1) / total_thread_num; // 스레드 당 처리할 행 개수 계산

    // 스레드를 저장할 벡터 생성
    vector<pthread_t> threads(total_thread_num);
    // 스레드 데이터를 저장할 벡터 생성
    vector<ThreadData> thread_data(total_thread_num);

    for (int t = 0; t < total_thread_num; ++t) {
        // 스레드가 처리할 행의 시작점
        int start = t * rows_per_thread;
        // 스레드가 처리할 행의 끝점
        int end = min(start + rows_per_thread, total_rows);

        // 스레드 데이터 초기화
        thread_data[t].start_row = start;
        thread_data[t].end_row = end;
        thread_data[t].A = &A;
        thread_data[t].B = &B;
        thread_data[t].C = &C;
        thread_data[t].multiplyKT = multiplyKT;

        // 스레드 생성
        pthread_create(&threads[t], nullptr, thread_matmul, &thread_data[t]);
    }

    // 모든 스레드가 종료될 때 까지 대기
    for (int t = 0; t < total_thread_num; ++t) {
        pthread_join(threads[t], nullptr);
    }
}

int main(int argc, char* argv[]) {
    // 프로그램 실행 시 인자 개수 확인
    if (argc < 5) {
        cerr << "Usage: ./attention_mp <shm_id> <out_rows> <out_cols> <head_index>\n";
        return 1;
    }

    // 명령행 인자 저장
    int shm_id = stoi(argv[1]);
    int out_rows = stoi(argv[2]);
    int out_cols = stoi(argv[3]);
    int head_index = stoi(argv[4]);

    // 공유 메모리 연결
    int* shm_ptr = reinterpret_cast<int*>(shmat(shm_id, nullptr, 0));
    if (shm_ptr == (void*)-1) {
        perror("shmat");
        return 1;
    }

    // 행렬 데이터를 저장할 변수 선언
    int q_r, q_s, k_r, k_s, v_r, v_s;
    // Q 행렬 입력 받기
    cin >> q_r >> q_s;
    vector<vector<int>> Q = input_matrix(q_r, q_s);
    // K 행렬 입력 받기
    cin >> k_r >> k_s;
    vector<vector<int>> K = input_matrix(k_r, k_s);
    // V 행렬 입력 받기
    cin >> v_r >> v_s;
    vector<vector<int>> V = input_matrix(v_r, v_s);

    // Q * K^T 계산
    vector<vector<int>> QK_T(q_r, vector<int>(k_r, 0));
    parallel_matmul(Q, K, QK_T, true);

    // (QK^T) * V 계산
    vector<vector<int>> output(q_r, vector<int>(v_s, 0));
    parallel_matmul(QK_T, V, output, false);

    // 공유 메모리에 결과 저장
    int* result_ptr = shm_ptr;
    long long offset = static_cast<long long>(head_index) * out_rows * out_cols;
    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            result_ptr[offset + i * out_cols + j] = output[i][j];
        }
    }

    // 공유 메모리 분리
    shmdt(shm_ptr);
    return 0;
}