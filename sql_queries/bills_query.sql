SELECT b.bill_id, e.earliest_bvid, b.session_year, b.session_num, measure_type, e.urgency, datediff(e.earliest_date,sd.start_date) as days_since_start, e.appropriation, e.vote_required, e.taxlevy, e.fiscal_committee, b.passed
            FROM bill_tbl b
            left join (select earliest.bill_id, earliest.earliest_date, bv.bill_version_id as earliest_bvid, bv.urgency, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee from
        				(select bill_id, min(bill_version_action_date) as earliest_date from bill_version_tbl
        				group by bill_id) earliest
        				join bill_version_tbl bv on (earliest.bill_id=bv.bill_id and earliest.earliest_date=bv.bill_version_action_date)) e
            on b.bill_id=e.bill_id
            join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
            where b.measure_type in ('AB' , 'SB') and b.session_year < "2015"