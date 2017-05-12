select avg(datediff(passed_date, intro_date)) as time_to_pass from
	(select only_passed.bill_id,  min(bill_version_action_date) as intro_date from
	(SELECT *
		FROM bill_tbl b
		where measure_type in ('AB' , 'SB') and passed=1) only_passed
	left join bill_version_tbl bv on only_passed.bill_id=bv.bill_id
	where bill_version_action = 'Introduced'
	group by only_passed.bill_id) intro_t
join
	(select only_passed.bill_id,  min(bill_version_action_date) as passed_date from
	(SELECT *
		FROM bill_tbl b
		where measure_type in ('AB' , 'SB') and passed=1) only_passed
	left join bill_version_tbl bv on only_passed.bill_id=bv.bill_id
	where bill_version_action = 'Enrolled'
	group by only_passed.bill_id) enrolled_t
on intro_t.bill_id=enrolled_t.bill_id
