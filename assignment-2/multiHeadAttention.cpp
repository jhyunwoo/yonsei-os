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