MKDIR   := mkdir -p
RMDIR   := rm -rf
CC      := g++
BUILD   := ./build
SRC     := ./src
INCLUDE := ./include
SRCS    := $(wildcard $(SRC)/*.cpp)
OBJS    := $(patsubst $(SRC)/%.cpp,$(BUILD)/%.o,$(SRCS))
EXE     := $(BUILD)/main
CFLAGS  := -I$(SRC) -I$(INCLUDE)  # Include both src and include directories

.PHONY: all run clean

all: $(EXE)

$(EXE): $(OBJS) | $(BUILD)
	$(CC) $^ -o $@

$(BUILD)/%.o: $(SRC)/%.cpp | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	$(MKDIR) $@

run: $(EXE)
	./$(EXE)

clean:
	$(RMDIR) $(BUILD)
