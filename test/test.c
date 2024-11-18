#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include "julia_init.h"

int main(int argc, char *argv[]) {
    // 加载 DubCmp 库
    void* handle = dlopen("/Users/tangjianfeng/code/julia_work/dubcmp/build/dubcmp/lib/libDubCmp.dylib", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "Unable to load library: %s\n", dlerror());
        return 1;
    }

    // 获取函数指针
    const char* (*greet)(void);
    *(void **)(&greet) = dlsym(handle, "greet");

    // 检查是否成功获取函数指针
    if (!greet) {
        fprintf(stderr, "Unable to find function: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }

    // 初始化 Julia
    init_julia(argc, argv);

    // 调用 greet 函数
    const char* greet_msg = greet();

    // 检查返回值
    if (greet_msg == NULL) {
        fprintf(stderr, "Error: greet returned a null pointer!\n");
    } else {
        printf("%s\n", greet_msg);
    }

    // 关闭 Julia
    shutdown_julia(0);

    // 关闭库
    dlclose(handle);
    return 0;
}
