select bv.bill_id, count(l.author_name) as n_authors from bill_version_authors_tbl bva
left join legislator_tbl l on bva.name=l.author_name and l.session_year=bva.session_year
join bill_version_tbl bv on bv.bill_version_id=bva.bill_version_id
where contribution='LEAD_AUTHOR' and bva.bill_version_id like '%INT' and (bv.bill_id like '%AB%' or bv.bill_id like '%SB%')
group by bill_id
order by count(*) desc