CC = gcc
CFLAGS = -Wall -g
BUILD_PATH = /Users/tangjianfeng/code/julia_work/dubcmp/build
LDFLAGS = -L$(BUILD_PATH)/dubcmp/lib -L$(BUILD_PATH)/dubcmp/lib/julia -ldl -lDubCmp -ljulia -rpath $(BUILD_PATH)/dubcmp/lib -rpath $(BUILD_PATH)/dubcmp/lib/julia

# 指定包含头文件和库文件的路径
LIB_PATH = $(BUILD_PATH)/dubcmp/lib:$(BUILD_PATH)/dubcmp/lib/julia
INCLUDE_PATH = $(BUILD_PATH)/dubcmp/include

# 输出文件名
TARGET = test

# 源文件
SRC = test.c

# 编译命令
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -I$(INCLUDE_PATH) -o $(TARGET) $(SRC) $(LDFLAGS)

# 运行命令，设置 DYLD_LIBRARY_PATH 确保加载动态库
run: $(TARGET)
	# 设置环境变量 DYLD_LIBRARY_PATH 来找到库文件
	DYLD_LIBRARY_PATH=$(LIB_PATH) ./$(TARGET)

# 清理编译生成的文件
clean:
	rm -f $(TARGET)
