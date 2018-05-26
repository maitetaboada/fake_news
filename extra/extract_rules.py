import pandas as pd


def filtered_rules(rules, rule_type):

    rules = rules.replace("[", "").replace("]", "").split(",")
    lexicalized_rules = []
    unlexicalized_rules = []
    for rule in rules:
        right_side = rule.split("-> ")
        if len(right_side) > 1:
            right_rule = right_side[1]
            if "'" in right_rule:
                lexicalized_rules.append(rule)
            else:
                unlexicalized_rules.append(rule)

    if rule_type == "lexicalized":
        return lexicalized_rules
    return unlexicalized_rules


def main():
    input_file = input("Please enter the input file name: ")
    output_file = input("Please enter the output file name: ")

    rules = pd.DataFrame.from_csv(input_file, index_col=None)
    rules["lexicalized_rules"] = rules["rules"].apply(lambda x: filtered_rules(x, "lexicalized"))
    rules["unlexicalized_rules"] = rules["rules"].apply(lambda x: filtered_rules(x, "unlexicalized"))
    rules.to_csv(output_file)


if __name__ == "__main__":
    main()
