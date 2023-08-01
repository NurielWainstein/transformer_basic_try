from datetime import datetime, timedelta

def generate_date_tuples():
    date_tuples = []
    current_date = datetime(1950, 1, 1)
    end_date = datetime(2051, 12, 1)

    while current_date < end_date:
        month_year_str = current_date.strftime('%m/%y')
        first_day_month_year_str = current_date.replace(day=1).strftime('%d/%m/%y')
        date_tuples.append((month_year_str, first_day_month_year_str))
        current_date += timedelta(days=31)
    return date_tuples

def separate_x_y(data):
    x_values = []
    y_values = []
    for x, y in data:
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values

date_tuples = generate_date_tuples()
x, y = separate_x_y(date_tuples)
print(x)
print(y)