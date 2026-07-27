.PHONY: all run1 run2 run3 clean

all: run1 run2 run3

run1:
	$(MAKE) -C run1

run2:
	@if [ -f run2/Makefile ]; then $(MAKE) -C run2; fi

run3:
	@if [ -f run3/Makefile ]; then $(MAKE) -C run3; fi

clean:
	$(MAKE) -C run1 clean
	@if [ -f run2/Makefile ]; then $(MAKE) -C run2 clean; fi
	@if [ -f run3/Makefile ]; then $(MAKE) -C run3 clean; fi
