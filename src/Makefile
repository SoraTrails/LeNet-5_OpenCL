OUTPUT_PATH=build
FLAGS= -std=c++11 -Wall -O3 -lOpenCL ${DEBUG}
PREDICT_FLAGS= -O3 -std=c++11 -DTEST_PREDICT -lopencv_core -lopencv_imgproc -lopencv_highgui -lOpenCL ${DEBUG}

# ifdef DEBUG
# 	${FLAGS} += -g
# 	${PREDICT_FLAGS} += -g
# endif

src = $(wildcard *.cpp)
target = $(patsubst %.cpp, ${OUTPUT_PATH}/%.o, ${src})

all:../Release/OpenCL_CNN ../Release/OpenCL_Predict_CNN

../Release/OpenCL_CNN: ${target}
	g++ -o ../Release/OpenCL_CNN ${target} ${FLAGS}
../Release/OpenCL_Predict_CNN: ${target}
	g++ ${PREDICT_FLAGS} -c -o ${OUTPUT_PATH}/main.o main.cpp 
	g++ -o ../Release/OpenCL_Predict_CNN ${target} ${PREDICT_FLAGS} 


${target}: ${OUTPUT_PATH}/%.o:%.cpp cnn.h
	@echo "compiling $@"
	g++ ${FLAGS} -c -o $@ $< 

.PHONY: clean

clean:
	rm build/*
	rm ../Release/OpenCL_CNN ../Release/OpenCL_Predict_CNN