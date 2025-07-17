OCAMLC=ocamlc
OCAMLFLAGS=-g

MODULES=math nn data main
CMOS=$(MODULES:=.cmo)

EXEC=diabetes_nn

all: $(EXEC)

%.cmo: %.ml
	$(OCAMLC) $(OCAMLFLAGS) -c $<

$(EXEC): $(CMOS)
	$(OCAMLC) $(OCAMLFLAGS) -o $(EXEC) $(CMOS)

clean:
	rm -f *.cmo *.cmi *.o $(EXEC)

.PHONY: all clean 