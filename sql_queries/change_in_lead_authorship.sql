SELECT 
    sum(sames)/count(sames), count(sames)
FROM
    (SELECT 
        t.bill_id,
            t.inames,
            t.pnames,
            (CASE
                WHEN INSTR(pnames, inames) THEN 1
                ELSE 0
            END) AS sames
    FROM
        (SELECT 
        introduced.bill_id,
            introduced.names AS inames,
            passed.names AS pnames
    FROM
        (SELECT 
        bv.bill_id, GROUP_CONCAT(bva.name) AS names
    FROM
        bill_version_authors_tbl bva
    JOIN legislator_tbl l ON bva.name = l.author_name
        AND l.session_year = bva.session_year
    JOIN bill_version_tbl bv ON bv.bill_version_id = bva.bill_version_id
    WHERE
        contribution = 'LEAD_AUTHOR'
            AND bv.bill_version_id LIKE '%INT'
    GROUP BY bv.bill_id) introduced
    JOIN (SELECT 
        b.bill_id, GROUP_CONCAT(bva.name) AS names
    FROM
        bill_tbl b
    JOIN bill_version_authors_tbl bva ON b.latest_bill_version_id = bva.bill_version_id
    WHERE
        passed = 1
            AND measure_type IN ('AB' , 'SB')
    GROUP BY b.bill_id) passed ON introduced.bill_id = passed.bill_id) t) agg
