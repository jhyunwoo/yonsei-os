#include <iostream>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstring>

using namespace std;

// 행렬 입력 함수
vector<vector<int>> input_matrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            cin >> matrix[i][j];
    return matrix;
}

// 행렬 출력 함수
void print_matrix(const vector<vector<int>>& mat) {
    for (const auto& row : mat) {
        for (int val : row)
            cout << val << " ";
        cout << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./multiHeadAttention <total_process_num>" << endl;
        return 1;
    }

    int total_process_num = atoi(argv[1]);

    int num_heads;
    cin >> num_heads;

    if (total_process_num != num_heads) {
        cerr << "Error: total_process_num must be equal to num_heads." << endl;
        return 1;
    }

    // HeadData 구조체
    struct HeadData {
        int q_r, q_s, k_r, k_s, v_r, v_s;
        vector<vector<int>> Q, K, V;
    };
    vector<HeadData> heads(num_heads);

    for (int h = 0; h < num_heads; ++h) {
        HeadData& hd = heads[h];
        cin >> hd.q_r >> hd.q_s;
        hd.Q = input_matrix(hd.q_r, hd.q_s);
        cin >> hd.k_r >> hd.k_s;
        hd.K = input_matrix(hd.k_r, hd.k_s);
        cin >> hd.v_r >> hd.v_s;
        hd.V = input_matrix(hd.v_r, hd.v_s);
    }

    // 동일 차원 가정
    int q_r = heads[0].q_r;
    int v_s = heads[0].v_s;

    // 공유 메모리 생성: 시간 정보 + 모든 결과 합
    int shm_size = (num_heads + q_r * v_s * num_heads) * sizeof(int);
    int shm_id = shmget(IPC_PRIVATE, shm_size, IPC_CREAT | 0666);
    int* shm_ptr = (int*)shmat(shm_id, NULL, 0);

    // 프로세스 생성
    for (int h = 0; h < num_heads; ++h) {
        int pipefd[2];
        pipe(pipefd);

        pid_t pid = fork();
        if (pid == 0) {
            dup2(pipefd[0], 0);
            close(pipefd[1]);

            string shm_id_str = to_string(shm_id);
            string row_str = to_string(q_r);
            string col_str = to_string(v_s);
            string head_str = to_string(h);
            string total_str = to_string(num_heads);

            execlp("./attention_mp", "attention_mp",
                   shm_id_str.c_str(), row_str.c_str(),
                   col_str.c_str(), head_str.c_str(), total_str.c_str(), NULL);
            perror("exec failed");
            exit(1);
        } else {
            close(pipefd[0]);
            FILE* w = fdopen(pipefd[1], "w");
            HeadData& hd = heads[h];

            // Q
            fprintf(w, "%d %d\n", hd.q_r, hd.q_s);
            for (const auto& row : hd.Q)
                for (int x : row) fprintf(w, "%d ", x);
            fprintf(w, "\n");

            // K
            fprintf(w, "%d %d\n", hd.k_r, hd.k_s);
            for (const auto& row : hd.K)
                for (int x : row) fprintf(w, "%d ", x);
            fprintf(w, "\n");

            // V
            fprintf(w, "%d %d\n", hd.v_r, hd.v_s);
            for (const auto& row : hd.V)
                for (int x : row) fprintf(w, "%d ", x);
            fprintf(w, "\n");

            fclose(w);
        }
    }

    // 모든 자식 대기
    for (int h = 0; h < num_heads; ++h)
        wait(NULL);

    // 결과 처리
    int* time_ptr = shm_ptr;
    int* result_ptr = shm_ptr + num_heads;

    int total_time = 0;
    for (int h = 0; h < num_heads; ++h)
        total_time += time_ptr[h];

    cout << total_time << endl;

    vector<vector<int>> final_result(q_r, vector<int>(v_s, 0));
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < q_r; ++i) {
            for (int j = 0; j < v_s; ++j) {
                final_result[i][j] += result_ptr[h * q_r * v_s + i * v_s + j];
            }
        }
    }

//    print_matrix(final_result);

    // 공유 메모리 해제
    shmdt(shm_ptr);
    shmctl(shm_id, IPC_RMID, NULL);

    return 0;
}

//#include <iostream>
//#include <vector>
//#include <unistd.h>
//#include <sys/wait.h>
//#include <sys/ipc.h>
//#include <sys/shm.h>
//#include <cstring>
//
//using namespace std;
//
//// 행렬 입력 함수
//vector<vector<int>> input_matrix(int rows, int cols) {
//    vector<vector<int>> matrix(rows, vector<int>(cols));
//    for (int i = 0; i < rows; ++i) {
//        for (int j = 0; j < cols; ++j) {
//            cin >> matrix[i][j];
//        }
//    }
//    return matrix;
//}
//
//// 행렬 출력 함수
//void print_matrix(const vector<vector<int>>& mat) {
//    for (const auto& row : mat) {
//        for (int val : row) {
//            cout << val << " ";
//        }
//        cout << "\n";
//    }
//}
//
//int main(int argc, char* argv[]) {
//    ios::sync_with_stdio(false);
//    cin.tie(nullptr);
//
//    if (argc < 2) {
//        cerr << "Usage: ./multiHeadAttention <total_process_num>\n";
//        return 1;
//    }
//
//    int total_process_num = atoi(argv[1]);
//
//    int num_heads;
//    cin >> num_heads;
//
//    if (total_process_num != num_heads) {
//        cerr << "Error: total_process_num must be equal to num_heads.\n";
//        return 1;
//    }
//
//    // HeadData 구조체
//    struct HeadData {
//        int q_r, q_s, k_r, k_s, v_r, v_s;
//        vector<vector<int>> Q, K, V;
//    };
//
//    vector<HeadData> heads(num_heads);
//
//    // 모든 HeadData 입력
//    for (int h = 0; h < num_heads; ++h) {
//        HeadData& hd = heads[h];
//        cin >> hd.q_r >> hd.q_s;
//        hd.Q = input_matrix(hd.q_r, hd.q_s);
//
//        cin >> hd.k_r >> hd.k_s;
//        hd.K = input_matrix(hd.k_r, hd.k_s);
//
//        cin >> hd.v_r >> hd.v_s;
//        hd.V = input_matrix(hd.v_r, hd.v_s);
//    }
//
//    // 모든 헤드는 동일 q_r, v_s라 가정
//    int q_r = heads[0].q_r;
//    int v_s = heads[0].v_s;
//
//    // -----------------------------------------------------------
//    // 공유 메모리 생성:
//    //  [0..(num_heads-1)] 구간: 각 헤드별 걸린 시간(밀리초)
//    //  [num_heads..(num_heads + q_r*v_s*num_heads - 1)] 구간:
//    //       각 헤드별 (q_r x v_s) 결과 행렬
//    // -----------------------------------------------------------
//    int shm_size = (num_heads + q_r * v_s * num_heads) * sizeof(int);
//    int shm_id = shmget(IPC_PRIVATE, shm_size, IPC_CREAT | 0666);
//    if (shm_id < 0) {
//        perror("shmget");
//        return 1;
//    }
//
//    // 부모에서 바로 attach (나중에 자식에서도 attach)
//    int* shm_ptr = (int*)shmat(shm_id, nullptr, 0);
//    if (shm_ptr == (void*)-1) {
//        perror("shmat");
//        return 1;
//    }
//
//    // 자식 프로세스 생성
//    vector<pid_t> pids(num_heads);
//
//    for (int h = 0; h < num_heads; ++h) {
//        // 파이프 준비
//        int pipefd[2];
//        if (pipe(pipefd) == -1) {
//            perror("pipe");
//            return 1;
//        }
//
//        pid_t pid = fork();
//        if (pid < 0) {
//            perror("fork");
//            return 1;
//        }
//
//        if (pid == 0) {
//            // 자식 프로세스
//            // 파이프 읽기 전용으로 연결
//            dup2(pipefd[0], STDIN_FILENO);
//            close(pipefd[0]);
//            close(pipefd[1]);
//
//            // attention_mp에 필요한 인자 세팅
//            // argv: [shm_id, out_rows, out_cols, head_index, num_heads]
//            execlp("./attention_mp",
//                   "attention_mp",
//                   to_string(shm_id).c_str(),  // 공유 메모리 ID
//                   to_string(q_r).c_str(),     // out_rows
//                   to_string(v_s).c_str(),     // out_cols
//                   to_string(h).c_str(),       // head_index
//                   to_string(num_heads).c_str(), // num_heads
//                   (char*)nullptr);
//
//            // execlp 실패 시:
//            perror("exec failed");
//            _exit(1);
//        }
//        else {
//            // 부모 프로세스
//            pids[h] = pid;
//
//            // 파이프 쓰기 전용으로 연결
//            close(pipefd[0]);
//            FILE* w = fdopen(pipefd[1], "w");
//            if (!w) {
//                perror("fdopen");
//                return 1;
//            }
//
//            // h번 HeadData에 대한 Q, K, V를 자식에게 전송
//            // 형식:
//            //   q_r q_s
//            //   Q 내용 ...
//            //   k_r k_s
//            //   K 내용 ...
//            //   v_r v_s
//            //   V 내용 ...
//            // (주의) 행렬 원소를 공백/개행으로 구분 (텍스트)
//            const auto& hd = heads[h];
//
//            fprintf(w, "%d %d\n", hd.q_r, hd.q_s);
//            for (const auto& row : hd.Q) {
//                for (int x : row) {
//                    fprintf(w, "%d ", x);
//                }
//            }
//            fprintf(w, "\n");
//
//            fprintf(w, "%d %d\n", hd.k_r, hd.k_s);
//            for (const auto& row : hd.K) {
//                for (int x : row) {
//                    fprintf(w, "%d ", x);
//                }
//            }
//            fprintf(w, "\n");
//
//            fprintf(w, "%d %d\n", hd.v_r, hd.v_s);
//            for (const auto& row : hd.V) {
//                for (int x : row) {
//                    fprintf(w, "%d ", x);
//                }
//            }
//            fprintf(w, "\n");
//
//            fclose(w);
//        }
//    }
//
//    // 모든 자식 프로세스 wait
//    for (int h = 0; h < num_heads; ++h) {
//        waitpid(pids[h], nullptr, 0);
//    }
//
//    // ------------------------------------
//    // 공유 메모리로부터 결과 수집
//    //   time_ptr[h]: h번째 head의 연산시간(밀리초)
//    //   result_ptr: h번째 (q_r x v_s) 행렬의 시작 오프셋
//    // ------------------------------------
//    int* time_ptr = shm_ptr;
//    int* result_ptr = shm_ptr + num_heads;
//
//    // (1) 총 시간 합산
//    long long total_time = 0;
//    for (int h = 0; h < num_heads; ++h) {
//        total_time += time_ptr[h];
//    }
//    cout << total_time << "\n";
//
//    // (2) multi-head 결과 합
//    vector<vector<int>> final_result(q_r, vector<int>(v_s, 0));
//    // 오프셋 = h * (q_r * v_s)
//    for (int h = 0; h < num_heads; ++h) {
//        int* head_result_ptr = result_ptr + (long long)h * q_r * v_s;
//        for (int i = 0; i < q_r; ++i) {
//            auto& rowFinal = final_result[i];
//            for (int j = 0; j < v_s; ++j) {
//                rowFinal[j] += head_result_ptr[i * v_s + j];
//            }
//        }
//    }
//
//    // 최종 결과 출력
////    print_matrix(final_result);
//
//    // 공유 메모리 해제
//    shmdt(shm_ptr);
//    shmctl(shm_id, IPC_RMID, nullptr);
//
//    return 0;
//}