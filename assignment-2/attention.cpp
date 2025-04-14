#include <iostream>
#include <vector>
#include <pthread.h>
#include <chrono>
#include <algorithm>

using namespace std;

int total_thread_num; // 총 스레드 수

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

// 스레드에 전달할 구조체
struct ThreadData {
    int start_row;
    int end_row;
    const vector<vector<int>>* A;
    const vector<vector<int>>* B;
    vector<vector<int>>* C;
    bool multiplyKT;
};

// 스레드에서 행렬 곱셈을 수행할 함수
void* thread_matmul(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);

    // A의 행
    int start = data->start_row;
    int end   = data->end_row;

    // A의 열 개수
    int A_cols = data->A->at(0).size();
    // B의 행 개수
    int B_rows = data->B->size();
    // B의 열 개수
    int B_cols = data->B->at(0).size();

    // 만약 전치 후 계산을 해야하는 경우
    if (data->multiplyKT) {
        for (int i = start; i < end; i++) {
            for (int j = 0; j < B_rows; j++) {
                int sum = 0;
                for (int x = 0; x < A_cols; x++) {
                    // 전치 후 행렬의 곱셉 수행
                    sum += (*data->A)[i][x] * (*data->B)[j][x];
                }
                // 결과 저장
                (*data->C)[i][j] = sum;
            }
        }
    } else { // 전치 없이 계산하는 경우
        for (int i = start; i < end; i++) {
            for (int x = 0; x < A_cols; x++) {
                int tmp = (*data->A)[i][x]; // tmp 변수에 A[i][x] 저장
                const auto& bRow = (*data->B)[x]; // B의 x번째 행
                auto& cRow = (*data->C)[i]; // 결과 행렬의 i번째 행
                for (int j = 0; j < B_cols; j++) {
                    // 곱셈 결과 저장
                    cRow[j] += tmp * bRow[j];
                }
            }
        }
    }
    return nullptr;
}

// 스레드를 사용해 병렬로 행렬 곱셈을 하기 위한 함수
void parallel_matmul(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, bool multiplyKT)
{
    // 계산 할 행렬의 크기 저장
    int total_rows = A.size();

    // 스레드 할당
    vector<pthread_t> threads(total_thread_num);
    vector<ThreadData> thread_data(total_thread_num);

    // 스레드를 최대한 많이 활용하기 위해 스레드 당 몇 줄의 행을 처리할지 계산
    int rows_per_thread = (total_rows + total_thread_num - 1) / total_thread_num;

    // 스레드를 total_thread_num 만큼 생성
    for (int t = 0; t < total_thread_num; t++) {
        // 계산을 시작할 행과 끝 행을 계산
        int start = t * rows_per_thread;
        int end   = min(start + rows_per_thread, total_rows);
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
    // 모든 스레드가 종료될 때까지 대기
    for (int t = 0; t < total_thread_num; t++) {
        pthread_join(threads[t], nullptr);
    }
}

// 행렬 출력 함수
void print_matrix(const vector<vector<int>>& mat) {
    for (const auto& row : mat) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << "\n";
    }
}

int main(int argc, char* argv[]){
    // 시간 측정 시작
    auto start_time = chrono::steady_clock::now();

    // 실행 할 때 받은 인자가 올바른지 확인
    if (argc < 2) {
        cerr << "Usage: ./program <thread_count>\n";
        return 1;
    }

    // 총 스레드 수 저장
    total_thread_num = stoi(argv[1]);

    // Q, K, V 행렬 정보 저장하는 변수
    int q_r, q_s, k_r, k_s, v_r, v_s;

    // Q 행렬 입력
    cin >> q_r >> q_s;
    vector<vector<int>> Q = input_matrix(q_r, q_s);

    // K 행렬 입력
    cin >> k_r >> k_s;
    vector<vector<int>> K = input_matrix(k_r, k_s);

    // V 행렬 입력
    cin >> v_r >> v_s;
    vector<vector<int>> V = input_matrix(v_r, v_s);

    // QK_T를 계산한 결과를 저장하는 변수
    vector<vector<int>> QK_T(q_r, vector<int>(k_r, 0));
    // 최종 결과를 저장하는 변수
    vector<vector<int>> attention(q_r, vector<int>(v_s, 0));

    // Q * K^T 계산
    parallel_matmul(Q, K, QK_T, true);

    // (QK^T) * V 계산
    parallel_matmul(QK_T, V, attention, false);

    // 시간 측정 종료
    auto end_time = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    // 결과 출력
    cout << duration << "\n";
    print_matrix(attention);

    return 0;
}