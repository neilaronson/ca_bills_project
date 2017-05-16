select t.bill_id, count(t.earliest_bvid) as ct from (SELECT 
    earliest.bill_id,
    bv.bill_version_id AS earliest_bvid,
    bill_xml,
    passed
FROM
    (SELECT 
        bill_id, MIN(bill_version_action_date) AS earliest_date
    FROM
        bill_version_tbl
    GROUP BY bill_id) earliest
        JOIN
    bill_version_tbl bv ON (earliest.bill_id = bv.bill_id
        AND earliest.earliest_date = bv.bill_version_action_date)
        JOIN
    bill_tbl b ON earliest.bill_id = b.bill_id
WHERE
	b.session_year < '2015' and b.measure_type in ('AB', 'SB')) t
group by t.bill_id
order by ct desc