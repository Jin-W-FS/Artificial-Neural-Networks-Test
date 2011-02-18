CC=gcc
CFLAGS=-g
LDFLAGS=-lm

TARGET=test
OBJS=test.o NeuralLayer.o
LOGS=errors.log test_out.log

$(TARGET):$(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $(TARGET)

test.o:NeuralLayer.h

NeuralLayer.o:NeuralLayer.h

.PHONY:clean data

$(LOGS):$(TARGET)
	./$(TARGET) -e errors.log -i test_in.log -o test_out.log

data:$(LOGS)
	gnuplot -persist plot.cfg

clean:
	-rm $(OBJS) $(TARGET)
