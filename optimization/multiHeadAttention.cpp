#include <iostream>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstring>
#include <chrono>

using namespace std;

// 행렬 입력 함수
vector<vector<int>> input_matrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cin >> matrix[i][j];
        }
    }
    return matrix;
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

int main(int argc, char* argv[]) {
    // 시간 측정 시작
    auto start_time = chrono::steady_clock::now();

    // 프로그램 실행 인자가 정확한지 확인
    if (argc < 2) {
        cerr << "Usage: ./multiHeadAttention <total_process_num>\n";
        return 1;
    }

    // 총 프로세스 수 저장
    int total_process_num = atoi(argv[1]);

    // 계산해야 하는 행렬 세트 계수 저장
    int num_heads;
    cin >> num_heads;

    // 만약 프로세스 수와 헤드 수가 다르면 오류 출력
    if (total_process_num != num_heads) {
        cerr << "Error: total_process_num must be equal to num_heads.\n";
        return 1;
    }

    // HeadData 구조체
    struct HeadData {
        int q_r, q_s, k_r, k_s, v_r, v_s;
        vector<vector<int>> Q, K, V;
    };

    // 헤드 개수 만큼 HeadData 구조체 생성
    vector<HeadData> heads(num_heads);

    // 모든 HeadData에 정보 입력
    for (int h = 0; h < num_heads; ++h) {
        HeadData& hd = heads[h];
        // Q 행렬 입력
        cin >> hd.q_r >> hd.q_s;
        hd.Q = input_matrix(hd.q_r, hd.q_s);
        // K 행렬 입력
        cin >> hd.k_r >> hd.k_s;
        hd.K = input_matrix(hd.k_r, hd.k_s);
        // V 행렬 입력
        cin >> hd.v_r >> hd.v_s;
        hd.V = input_matrix(hd.v_r, hd.v_s);
    }

    // 행렬의 크기 저장
    int q_r = heads[0].q_r;
    int v_s = heads[0].v_s;

    // 공유 메모리 크기 계산
    int shm_size = q_r * v_s * num_heads * sizeof(int);
    // 공유 메모리 생성
    int shm_id = shmget(IPC_PRIVATE, shm_size, IPC_CREAT | 0666);
    // 공유 메모리 생성 실패시 프로그램 종료
    if (shm_id < 0) {
        perror("shmget");
        return 1;
    }

    // 부모 프로세스에 공유 메모리 연결
    int* shm_ptr = (int*)shmat(shm_id, nullptr, 0);
    if (shm_ptr == (void*)-1) {
        perror("shmat");
        return 1;
    }

    // 자식 프로세스 생성
    vector<pid_t> pids(num_heads);

    for (int h = 0; h < num_heads; ++h) {
        // 각 프로세스에 데이터 전달을 위한 파이프 생성
        int pipefd[2];
        if (pipe(pipefd) == -1) {
            perror("pipe");
            return 1;
        }

        // 자식 프로세스 생성
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }

        if (pid == 0) { // 자식 프로세스
            // 파이프 읽기 전용으로 연결
            dup2(pipefd[0], STDIN_FILENO);
            close(pipefd[0]);
            close(pipefd[1]);

            // attention_mp 실행
            execlp("./attention_mp",
                   "attention_mp",
                   to_string(shm_id).c_str(),
                   to_string(q_r).c_str(),
                   to_string(v_s).c_str(),
                   to_string(h).c_str(),
                   (char*)nullptr);

            // execlp 실패 시:
            perror("exec failed");
            _exit(1);
        }
        else { // 부모 프로세스
            pids[h] = pid;

            // 파이프를 쓰기 전용으로 연결
            close(pipefd[0]);
            FILE* w = fdopen(pipefd[1], "w");
            if (!w) {
                perror("fdopen");
                return 1;
            }

            // h번째 Q 행렬 정보를 자식에게 전송
            fprintf(w, "%d %d\n", heads[h].q_r, heads[h].q_s);
            for (const auto& row : heads[h].Q) {
                for (int x : row) {
                    fprintf(w, "%d ", x);
                }
            }
            fprintf(w, "\n");
            // h번째 K 행렬 정보를 자식에게 전송
            fprintf(w, "%d %d\n", heads[h].k_r, heads[h].k_s);
            for (const auto& row : heads[h].K) {
                for (int x : row) {
                    fprintf(w, "%d ", x);
                }
            }
            fprintf(w, "\n");
            // h번째 V 행렬 정보를 자식에게 전송
            fprintf(w, "%d %d\n", heads[h].v_r, heads[h].v_s);
            for (const auto& row : heads[h].V) {
                for (int x : row) {
                    fprintf(w, "%d ", x);
                }
            }
            fprintf(w, "\n");
            fclose(w);
        }
    }

    // 모든 자식 프로세스 대기
    for (int h = 0; h < num_heads; ++h) {
        waitpid(pids[h], nullptr, 0);
    }

    // 계산 결과 수집을 위한 공유 메모리 포인터
    int* result_ptr = shm_ptr;

    // 최종 결과 저장 변수
    vector<vector<int>> final_result(q_r, vector<int>(v_s, 0));
    // 각 프로세스의 결과 합산
    for (int h = 0; h < num_heads; ++h) {
        int* head_result_ptr = result_ptr + (long long)h * q_r * v_s;
        for (int i = 0; i < q_r; ++i) {
            for (int j = 0; j < v_s; ++j) {
                final_result[i][j] += head_result_ptr[i * v_s + j];
            }
        }
    }

    // 시간 측정 종료
    auto end_time = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    // 걸린 시간, 최종 결과 출력
    cout << duration << endl;
    print_matrix(final_result);

    // 공유 메모리 해제
    shmdt(shm_ptr);
    shmctl(shm_id, IPC_RMID, nullptr);

    return 0;
}