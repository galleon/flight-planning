from pybada.bada4 import BADA4_jet_CR, Table

print("BOOM !!!")

aircraft = BADA4_jet_CR("A330-341")

#       CCI| -0.000522  0.104419  0.544326  0.984234  1.424141  2.373568
#        CW|
# =======================================================================
#  1.323543|  0.500453  0.603029  0.777386  0.814373  0.832772  0.853201

try:
    aircraft.best_mach(10.0, 10.0)
except Exception as e:
    assert e.__class__.__name__ == "ValueError"
    print("BADA BOOM !!!")


assert aircraft.best_mach(-0.000522, 1.323543) == 0.500453

assert aircraft.best_mach(+2.373568, 5.696210) == 0.801180

assert (
    aircraft.best_mach(0.544326, 3.728510) == 0.815506
), f"{aircraft.best_mach(0.544326, 3.728510)} != 0.815506 [{100*round(aircraft.best_mach(0.544326, 3.728510)/0.815506 - 1, 2)}%]"

assert (
    aircraft.best_mach(2.373568, 1.323543) == 0.853201
), f"{aircraft.best_mach(2.373568, 1.323543)} != 0.853201[{100*round(aircraft.best_mach(2.373568, 1.323543)/0.853201 - 1, 2)}%]"

from tempfile import mkstemp

fd, path = mkstemp()

# use a context manager to open the file at that path and close it again
with open(path, "w") as f:
    f.write("Table name: NO_NAME\n")
    f.write("Table type: 2D\n")
    f.write("Table variables: X, Y\n")
    f.write("Table dimension: 5x2\n")
    f.write("    X| -2.0  -1.0  0.0  1.0  2.0\n")
    f.write("    Y|\n")
    f.write("================================\n")
    f.write(" 10.0| -7.0  -6.0  0.0  NaN  NaN\n")
    f.write(" 69.0| -9.0  -4.0  0.0  4.0  9.0\n")

table = Table(path)

assert table(-2.0, 10.0) == -7.0
assert table(-1.0, 10.0) == -6.0
