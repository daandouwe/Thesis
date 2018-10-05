from utils import add_dummy_tags, substitute_leaves


tagless_tree = "(S (S (NP Big investment banks) (VP refused (S (VP to (VP step))) up (PP to (NP the plate)) (S (VP to (VP support (NP the beleaguered floor traders)))) (PP by (S (VP buying (NP (NP big blocks) (PP of (NP stock)))))))) , (NP traders) (VP say) .)"
unked_tree = "(S (NP The UNK-LC-ing) (VP has (ADVP already) (VP begun)) .)"
new_leaves = "The fucking has already begun .".split()

# print(tagless_tree)
# print()
# print(add_dummy_tags(tagless_tree))

print(unked_tree)
print()
print(substitute_leaves(unked_tree, new_leaves))
