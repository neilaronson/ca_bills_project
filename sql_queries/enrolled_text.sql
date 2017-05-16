SELECT 
    latest.bill_id,
    bv.bill_version_id AS enrolled_bvid,
    bill_xml,
    passed
FROM
(SELECT 
        bill_id, MAX(bill_version_action_date) AS latest_date
    FROM
        bill_version_tbl
	WHERE
		bill_version_id like '%ENR'
    GROUP BY bill_id) latest
JOIN
bill_version_tbl bv ON (latest.bill_id = bv.bill_id
	AND latest.latest_date = bv.bill_version_action_date)
	JOIN
bill_tbl b ON latest.bill_id = b.bill_id