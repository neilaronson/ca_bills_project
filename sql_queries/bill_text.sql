SELECT 
    b.bill_id, b.latest_bill_version_id, bv.bill_xml, passed
FROM
    bill_tbl b
        JOIN
    bill_version_tbl bv ON b.latest_bill_version_id = bv.bill_version_id
WHERE
    b.session_year < '2015' and b.measure_type in ('AB', 'SB')