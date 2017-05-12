create view start_dates as
select session_year, session_num, min(action_date) as start_date from bill_history_tbl
group by session_year, session_num