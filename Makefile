CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm
TARGET = neural_network
SRC = src/main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
