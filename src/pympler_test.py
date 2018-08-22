import types

from pympler import muppy, tracker

tr = tracker.SummaryTracker()
# print(len(all_objects))
tr.print_diff()
tr.print_diff()
tr.print_diff()
all_objects = muppy.get_objects()
my_types = muppy.filter(all_objects, Type=list)
print(len(my_types))
tr.print_diff()

quit()


print(len(my_types))
tr.print_diff()

tr.print_diff()

def show(all_objects):
    for obj in all_objects:
        if type(obj) == list:
            print(obj)
    print(len(all_objects))
    print(tr.print_diff())

show(all_objects)
