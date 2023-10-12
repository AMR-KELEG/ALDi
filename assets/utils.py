def commasize_number(number):
    number_str = str(number)
    output_number = ""
    if len(number_str) % 3 == 1:
        output_number = number_str[0]
        number_str = number_str[1:]

    elif len(number_str) % 3 == 2:
        output_number = number_str[:2]
        number_str = number_str[2:]

    if number_str:
        if output_number:
            output_number = output_number + ","
        output_number += ",".join(
            [number_str[i : i + 3] for i in range(0, len(number_str), 3)]
        )

    return output_number
