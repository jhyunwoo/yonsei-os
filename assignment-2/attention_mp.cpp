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
        cerr << "Usage: ./attention_mp <shm_id> <out_rows> <out_cols> <head_index> <num_heads>" << endl;
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
//
//#include <iostream>
//#include <vector>
//#include <sys/ipc.h>
//#include <sys/shm.h>
//#include <cstdlib>
//#include <pthread.h>
//#include <chrono>
//
//using namespace std;
//
//static int total_thread_num = 2;
//
//struct ThreadData {
//    int start_row;
//    int end_row;
//    // Q*K^T용
//    const vector<vector<int>>* A; // Q (또는 중간 곱 결과)
//    const vector<vector<int>>* B; // K (전치 사용 없이 직접 접근)
//    vector<vector<int>>* C;       // 결과(QK^T)
//
//    // (QK^T)*V용
//    bool multiplyKT; // true 면 Q*K^T 계산, false 면 (QK^T)*V 계산
//    int common_dim;  // 실제로 곱셈에 사용되는 가운데 차원
//};
//
////--------------------------------------------------------------------------
//// 1) Q*K^T 계산용:
////    결과행렬 C의 (i, j)는 sum(A[i, x] * B[j, x])  (B는 원본 K로서 j행 x열을 접근)
//// 2) (QK^T)*V 계산용:
////    결과행렬 C의 (i, j)는 sum(A[i, x] * B[x, j])  (B는 V로서 x행 j열을 접근)
////
//// 루프 순서를 i, x, j 또는 i, j, x로 조정해 캐시 효율을 높이는 것이 핵심입니다.
////--------------------------------------------------------------------------
//
//void* thread_matmul(void* arg) {
//    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
//
//    int start = data->start_row;
//    int end   = data->end_row;
//
//    // A는 [start_row ~ end_row) x common_dim, B는 형식에 따라 접근
//    // C는 [start_row ~ end_row) 결과를 저장
//    const auto& A = *(data->A);
//    const auto& B = *(data->B);
//    auto&       C = *(data->C);
//
//    // 공통 곱셈 루프에서 가운데 도는 차원
//    int K = data->common_dim;
//
//    if (data->multiplyKT) {
//        // -------- Q*K^T 계산 --------
//        //  C[i][j] = Σ( A[i][x] * B[j][x] )
//        //  (B가 K라면, j행 x열 = K[j][x])
//        for (int i = start; i < end; i++) {
//            for (int j = 0; j < static_cast<int>(B.size()); j++) {
//                // B.size() == k_r (K의 행 수). QK^T 결과는 (q_r x k_r)
//                int sum = 0;
//                for (int x = 0; x < K; x++) {
//                    sum += A[i][x] * B[j][x];
//                }
//                C[i][j] = sum;
//            }
//        }
//    } else {
//        // -------- (QK^T)*V 계산 --------
//        //  C[i][j] = Σ( A[i][x] * B[x][j] )
//        //  여기서 A = QK^T, B = V
//        for (int i = start; i < end; i++) {
//            for (int x = 0; x < K; x++) {
//                // A[i][x]를 미리 꺼내두어 캐시 사용
//                int tmp = A[i][x];
//                // B[x] = V의 x번째 행
//                const auto& vRow = B[x];
//                auto& cRow = C[i];
//                for (int j = 0; j < static_cast<int>(cRow.size()); j++) {
//                    cRow[j] += tmp * vRow[j];
//                }
//            }
//        }
//    }
//    return nullptr;
//}
//
////--------------------------------------------------------------------------
//// (Q*K^T) 또는 (QK^T)*V 곱셈을 Pthread로 나눠서 수행
//// A, B, C, multiplyKT, common_dim을 설정한 뒤 쓰레드 생성
////--------------------------------------------------------------------------
//void parallel_matmul(
//    const vector<vector<int>>& A,
//    const vector<vector<int>>& B,
//    vector<vector<int>>& C,
//    bool multiplyKT,
//    int common_dim
//) {
//    int rowsA = A.size();
//
//    // 쓰레드당 할당할 행의 개수
//    int rows_per_thread = (rowsA + total_thread_num - 1) / total_thread_num;
//
//    vector<pthread_t> threads(total_thread_num);
//    vector<ThreadData> thread_data(total_thread_num);
//
//    for (int t = 0; t < total_thread_num; ++t) {
//        int start = t * rows_per_thread;
//        int end = min(start + rows_per_thread, rowsA);
//
//        thread_data[t].start_row   = start;
//        thread_data[t].end_row     = end;
//        thread_data[t].A           = &A;
//        thread_data[t].B           = &B;
//        thread_data[t].C           = &C;
//        thread_data[t].multiplyKT  = multiplyKT;
//        thread_data[t].common_dim  = common_dim;
//
//        pthread_create(&threads[t], nullptr, thread_matmul, &thread_data[t]);
//    }
//
//    for (int t = 0; t < total_thread_num; ++t) {
//        pthread_join(threads[t], nullptr);
//    }
//}
//
//int main(int argc, char* argv[]) {
//    if (argc < 6) {
//        cerr << "Usage: ./attention_mp <shm_id> <out_rows> <out_cols> <head_index> <num_heads>" << endl;
//        return 1;
//    }
//
//    int shm_id     = stoi(argv[1]);
//    int out_rows   = stoi(argv[2]);
//    int out_cols   = stoi(argv[3]);
//    int head_index = stoi(argv[4]);
//    int num_heads  = stoi(argv[5]);
//
//    // 공유 메모리 attach
//    int* shm_ptr = reinterpret_cast<int*>(shmat(shm_id, nullptr, 0));
//
//    // Q, K, V 행렬 읽기
//    int q_r, q_s, k_r, k_s, v_r, v_s;
//    cin >> q_r >> q_s;
//    vector<vector<int>> Q(q_r, vector<int>(q_s));
//    for (auto &row : Q) {
//        for (int &x : row) cin >> x;
//    }
//
//    cin >> k_r >> k_s;
//    vector<vector<int>> K(k_r, vector<int>(k_s));
//    for (auto &row : K) {
//        for (int &x : row) cin >> x;
//    }
//
//    cin >> v_r >> v_s;
//    vector<vector<int>> V(v_r, vector<int>(v_s));
//    for (auto &row : V) {
//        for (int &x : row) cin >> x;
//    }
//
//    // 측정 시작
//    auto start_time = chrono::steady_clock::now();
//
//    // --------------------------
//    // 1) Q*K^T 계산
//    //    Q (q_r x q_s), K (k_r x k_s)
//    //    여기서 q_s == k_s 라 가정
//    //    Q*K^T => 결과 (q_r x k_r)
//    //    => C[i][j] = sum_{x=0..q_s-1} Q[i][x]*K[j][x]
//    // --------------------------
//    // (주의) Q*K^T를 일반적인 matmul 함수로 돌리려면
//    // B를 실제로 전치해야 하지만,
//    // 여기서는 thread_matmul() 내부에서 j, x 순서를 바꿔 접근해
//    // 별도의 전치 배열 할당 없이 계산하도록 구성.
//    // --------------------------
//    vector<vector<int>> QK_T(q_r, vector<int>(k_r, 0));
//    parallel_matmul(Q, K, QK_T, /*multiplyKT=*/true, /*common_dim=*/q_s);
//
//    // --------------------------
//    // 2) (QK^T)*V
//    //    QK^T (q_r x k_r), V (v_r x v_s)
//    //    여기서 k_r == v_r 라 가정
//    //    => 결과 (q_r x v_s)
//    // --------------------------
//    vector<vector<int>> output(q_r, vector<int>(v_s, 0));
//    parallel_matmul(QK_T, V, output, /*multiplyKT=*/false, /*common_dim=*/k_r);
//
//    // 측정 종료
//    auto end_time = chrono::steady_clock::now();
//    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
//
//    // 공유메모리에 시간 기록
//    shm_ptr[head_index] = duration;
//
//    // 3) 최종 결과를 head_index 위치에 맞춰 저장
//    //    [num_heads]부터 시작해서 (out_rows*out_cols)씩 offset
//    int* result_ptr = shm_ptr + num_heads;
//    for (int i = 0; i < out_rows; ++i) {
//        for (int j = 0; j < out_cols; ++j) {
//            result_ptr[head_index * out_rows * out_cols + i * out_cols + j] = output[i][j];
//        }
//    }
//
//    // 공유메모리 detach
//    shmdt(shm_ptr);
//    return 0;
//}