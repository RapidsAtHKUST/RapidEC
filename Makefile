ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

volta:
	mkdir -p bin
	nvcc $(INC) $(LIB) -Iinclude -arch=compute_70 -code=sm_70 src/gsv.cu -o bin/gsv -lgmp

test: volta
	/usr/bin/time ./bin/gsv

debug:
	mkdir -p bin
	nvcc $(INC) $(LIB) -DDEBUG -Iinclude -arch=compute_70 -code=sm_70 src/gsv.cu -o bin/gsv -lgmp
	./bin/gsv

clean:
	rm -rf bin
