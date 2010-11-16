CC=gcc
CFLAGS=-lm
LDFLAGS=

TARGET=test
OBJS=test.o NeuralNode.o

$(TARGET):$(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $(TARGET)

clean:
	rm $(OBJS) $(TARGET)
