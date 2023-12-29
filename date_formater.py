from datetime import datetime, timedelta

def date_simulation(event_date, days=100):
    end_datetime = datetime.strptime(event_date, "%Y-%m-%d %H:%M:%S")
    # Start from the end datetime and go backwards
    current_datetime = end_datetime
    datetimes_list = []

    for day in range(days):
        for hour in range(24):
            datetimes_list.append(current_datetime - timedelta(days=day, hours=hour))
    
    # Sort the list to be in order from past to present
    datetimes_list.sort()
    return datetimes_list

def calculate_days_between(dates):
    date1, date2 = dates
    date1 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return (date2 - date1).days

def get_month(row):
    dt = datetime.strptime(row, "%Y-%m-%d %H:%M:%S")
    return dt.month

def get_day_of_week(row):
    dt = datetime.strptime(row, "%Y-%m-%d %H:%M:%S")
    return dt.weekday()
    # return day of week

def get_hour_of_day(row):
    dt = datetime.strptime(row, "%Y-%m-%d %H:%M:%S")
    return dt.hour

def create_date_inputs(publish_date,event_date):
    days_b4_event = calculate_days_between([publish_date,event_date])
    month = get_month(publish_date)
    week_day = get_day_of_week(publish_date)
    hour = get_hour_of_day(publish_date)
    return days_b4_event, month, week_day, hour

def to_str(row):
    return f'{str(row.day)}-{str(row.month)}-{str(row.year)} h:{str(row.hour)}' 