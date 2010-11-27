CC=gcc
CFLAGS=-g
LDFLAGS=-lm

TARGET=test
OBJS=test.o NeuralLayer.o

$(TARGET):$(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $(TARGET)

test.o:NeuralLayer.h

NeuralLayer.o:NeuralLayer.h


.PHONY:clean

clean:
	-rm $(OBJS) $(TARGET)
