#include <iostream>
#include <vector>
#include <pthread.h>
#include <chrono>

using namespace std;

// 전역 변수
int total_thread_num; // 총 스레드 수
vector<vector<int>> Q, K, V; // 입력 받을 행렬
vector<vector<int>> QK_T; // Q × K^T 행렬
vector<vector<int>> attention; // 계산 결과

// 행렬을 입력 받는 함수
vector<vector<int>> input_matrix(int rows, int cols) {
  // 행렬을 저장할 변수 선언
    vector<vector<int>> matrix(rows, vector<int>(cols));
    // 행렬 입력
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cin >> matrix[i][j];
        }
    }
    // 입력 받은 행렬 반환
    return matrix;
}

// 행렬을 transpose 하는 함수
vector<vector<int>> transpose(const vector<vector<int>>& mat) {
  // 행렬의 크기를 저장
    int rows = mat.size();
    int cols = mat[0].size();

    // 전치 행렬을 저장할 변수 선언
    vector<vector<int>> trans(cols, vector<int>(rows));
    // 전치 수행
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            trans[j][i] = mat[i][j];
        }
    }
    // 전치 행렬 반환
    return trans;
}

// 스레드 생성에 사용할 구조체
struct ThreadData {
    int start_row;
    int end_row;
    const vector<vector<int>>* A;
    const vector<vector<int>>* B;
    vector<vector<int>>* result;
};

// 스레드에서 matmul을 계산하는 함수
void* thread_matmul(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int a_cols = data->A->at(0).size();
    int b_cols = data->B->at(0).size();
    int start_row = data->start_row;
    int end_row = data->end_row;

    // 행렬 곱셈 계산
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            int sum = 0;
            for (int k = 0; k < a_cols; ++k) {
                sum += data->A->at(i)[k] * data->B->at(k)[j];
            }
            data->result->at(i)[j] = sum;
        }
    }
    return nullptr;
}

// 스레드를 사용하여 병렬 계산하는 함수
void parallel_matmul(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result) {
    // 스레드 생성
    vector<pthread_t> threads(total_thread_num);
    vector<ThreadData> thread_data(total_thread_num);

    int total_rows = A.size();
    int rows_per_thread = (total_rows + total_thread_num - 1) / total_thread_num; // 스레드 당 처리할 행 수

    for (int i= 0; i< total_thread_num; i++) {
        int start = i* rows_per_thread;
        int end = min(start + rows_per_thread, total_rows);
        thread_data[i] = { start, end, &A, &B, &result };
        pthread_create(&threads[i], nullptr, thread_matmul, &thread_data[i]);
    }

    for (int i= 0; i< total_thread_num; i++) {
        pthread_join(threads[i], nullptr);
    }
}

// 행렬 출력
void print_matrix(const vector<vector<int>>& mat) {
    for (const auto& row : mat) {
        for (int val : row)
            cout << val << " ";
        cout << endl;
    }
}

int main(int argc, char* argv[]){
    int q_r, q_s, k_r, k_s, v_r, v_s;
    total_thread_num = std::stoi(argv[1]);

    // Q Input
    cin >> q_r >> q_s;
    vector<vector<int>> Q(q_r, vector<int>(q_s));
        Q = input_matrix(q_r, q_s);
        // K Input
        cin >> k_r >> k_s;
        vector<vector<int>> K(k_r, vector<int>(k_s));
        K = input_matrix(k_r, k_s);
        // V Input
        cin >> v_r >> v_s;
        vector<vector<int>> V(v_r, vector<int>(v_s));
        V= input_matrix(v_r, v_s);

        auto start_time = chrono::steady_clock::now();

        vector<vector<int>> K_T = transpose(K);
        QK_T = vector<vector<int>>(q_r, vector<int>(k_r));
        attention = vector<vector<int>>(q_r, vector<int>(v_s));

        // Parallel Q × K^T
        parallel_matmul(Q, K_T, QK_T);

        // Parallel (QK^T) × V
        parallel_matmul(QK_T, V, attention);

        // 시간 측정 끝
        auto end_time = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

        // 출력
        cout << duration << endl;
//        print_matrix(attention);


    return 0;
}