SELECT 
    passed_table.session_year,
    passed_table.passed,
    all_table.all_count,
    passed_table.passed / all_table.all_count AS percent_passed
FROM
    (SELECT 
        session_year, COUNT(bill_id) AS passed
    FROM
        bill_tbl
    WHERE
        measure_type IN ('AB' , 'SB')
            AND measure_state IN ('Enrolled' , 'Chaptered')
    GROUP BY session_year) AS passed_table
        JOIN
    (SELECT 
        session_year, COUNT(bill_id) AS all_count
    FROM
        bill_tbl
    WHERE
        measure_type IN ('AB' , 'SB')
    GROUP BY session_year) AS all_table ON passed_table.session_year = all_table.session_year