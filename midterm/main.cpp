#include <iostream>
#include <vector>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>

int main()
{
    pid_t pid;
    printf("about to call fork\n");
    pid = fork();
    if(pid==0){
        printf("I'm a child\n");
    }else {
printf("I'm a parent.\n");
}
}
