.PHONY: all run1 run2 run3 run4 run5 run6 clean

all: run1 run2 run3 run4 run5 run6

run1:
	$(MAKE) -C run1

run2:
	@if [ -f run2/Makefile ]; then $(MAKE) -C run2; fi

run3:
	@if [ -f run3/Makefile ]; then $(MAKE) -C run3; fi

run4:
	@if [ -f run4/Makefile ]; then $(MAKE) -C run4; fi

run5:
	@if [ -f run5/Makefile ]; then $(MAKE) -C run5; fi

run6:
	@if [ -f run6/Makefile ]; then $(MAKE) -C run6; fi

clean:
	$(MAKE) -C run1 clean
	@if [ -f run2/Makefile ]; then $(MAKE) -C run2 clean; fi
	@if [ -f run3/Makefile ]; then $(MAKE) -C run3 clean; fi
	@if [ -f run4/Makefile ]; then $(MAKE) -C run4 clean; fi
	@if [ -f run5/Makefile ]; then $(MAKE) -C run5 clean; fi
	@if [ -f run6/Makefile ]; then $(MAKE) -C run6 clean; fi
