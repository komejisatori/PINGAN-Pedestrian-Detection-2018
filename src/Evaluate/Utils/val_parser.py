
with open('val_result.txt', 'w') as result, open('val_annotations.txt', 'r') as the_file:
    lines = the_file.readlines()
    for x, line in enumerate(lines):
        parts = line.split(" ")
        if len(parts) <= 1:
            print(parts)
        else:
            image = parts[0]
            count = int((len(parts) - 1) / 5)
            for i in range(count):
                result_line = parts[0]
                result_line += " "
                result_line += "1"
                for j in range(4):
                    result_line += " "
                    result_line += parts[5 * i + j + 2]
                result_line += "\n"

                result.write(result_line)

                result_line = parts[0]
