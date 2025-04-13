#include <iostream>
#include <vector>
#include <pthread.h>
#include <chrono>

using namespace std;

vector<vector<int>> input_matrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cin >> matrix[i][j];
        }
    }
    return matrix;
}

// 전치 행렬
vector<vector<int>> transpose(const vector<vector<int>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    vector<vector<int>> trans(cols, vector<int>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            trans[j][i] = mat[i][j];
    return trans;
}

// 행렬 곱
vector<vector<int>> matmul(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int a_rows = A.size(), a_cols = A[0].size();
    int b_rows = B.size(), b_cols = B[0].size();
    vector<vector<int>> result(a_rows, vector<int>(b_cols, 0));
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                result[i][j] += A[i][k] * B[k][j];
    return result;
}

// 행렬 출력
void print_matrix(const vector<vector<int>>& mat) {
    for (const auto& row : mat) {
        for (int val : row)
            cout << val << " ";
        cout << endl;
    }
}

int main(int total_thread_num){
    int q_r, q_s, k_r, k_s, v_r, v_s;

    for(int l=0; l<total_thread_num;l++){
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
        // Step 1: K^T
        vector<vector<int>> K_T = transpose(K);

        // Step 2: Q * K^T
        vector<vector<int>> QK_T = matmul(Q, K_T);

        // Step 3: (QK^T) * V
        vector<vector<int>> attention = matmul(QK_T, V);

        // 시간 측정 끝
        auto end_time = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

        // 출력
        cout << duration << endl;
        print_matrix(attention);
    }

    return 0;
}