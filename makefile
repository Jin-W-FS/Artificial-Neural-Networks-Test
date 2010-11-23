CC=gcc
CFLAGS=-g
LDFLAGS=-lm

TARGET=test
OBJS=test.o NeuralLayer.o

$(TARGET):$(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $(TARGET)

clean:
	rm $(OBJS) $(TARGET)
