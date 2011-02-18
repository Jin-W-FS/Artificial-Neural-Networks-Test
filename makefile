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

.PHONY:clean show_error show_test

$(LOGS):$(TARGET)
	./$(TARGET) -e errors.log -i test_in.log -o test_out.log

show_error:errors.log errors.plt
	gnuplot -persist errors.plt

show_test:test_in.log test_out.log tests.plt
	gnuplot -persist tests.plt

clean:
	-rm $(OBJS) $(TARGET)
