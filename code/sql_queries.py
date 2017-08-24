"""Contains all SQL queries used in DataPrep class"""

def intro_bills_query():
    return """SELECT b.bill_id, bv.bill_version_id, b.session_year, b.session_num, measure_type, bv.urgency, datediff(bv.bill_version_action_date,sd.start_date) as days_since_start, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee, b.passed
        FROM bill_tbl b
        left join bill_version_tbl bv
        on b.bill_id=bv.bill_id and bv.bill_version_id like '%INT'
        join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
        where b.measure_type in ('AB' , 'SB') and b.session_year < '2015'"""

def intro_bills_query_test():
    return """SELECT b.bill_id, bv.bill_version_id, b.session_year, b.session_num, measure_type, bv.urgency, datediff(bv.bill_version_action_date,sd.start_date) as days_since_start, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee, b.passed
        FROM bill_tbl b
        left join bill_version_tbl bv
        on b.bill_id=bv.bill_id and bv.bill_version_id like '%INT'
        join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
        where b.measure_type in ('AB' , 'SB') and b.session_year >= '2015'"""

def intro_bills_query_all():
    return """SELECT b.bill_id, bv.bill_version_id, b.session_year, b.session_num, measure_type, bv.urgency, datediff(bv.bill_version_action_date,sd.start_date) as days_since_start, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee, b.passed
        FROM bill_tbl b
        left join bill_version_tbl bv
        on b.bill_id=bv.bill_id and bv.bill_version_id like '%INT'
        join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
        where b.measure_type in ('AB' , 'SB')"""

def intro_authors_query():
    return """SELECT
            bv.bill_id, l.author_name, l.party, l.district, bva.session_year
        FROM
            bill_version_authors_tbl bva
                LEFT JOIN
            legislator_tbl l ON bva.name = l.author_name
                AND l.session_year = bva.session_year
                JOIN
            bill_version_tbl bv ON bv.bill_version_id = bva.bill_version_id
        WHERE
            contribution = 'LEAD_AUTHOR'

                AND bva.bill_version_id LIKE '%INT'
                AND (bv.bill_id LIKE '%AB%'
                OR bv.bill_id LIKE '%SB%')"""

def intro_earliest_version_query():
    return """select bv.bill_id, bv.bill_xml from bill_version_tbl bv
        where bv.bill_version_id like '%INT'
        """

def amd_bills_query():
    return """SELECT b.bill_id, bv.bill_version_id, b.session_year, b.session_num, measure_type, bv.urgency, datediff(bv.bill_version_action_date,sd.start_date) as days_since_start, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee, b.passed
        FROM bill_tbl b
        join bill_version_tbl bv
        on b.bill_id=bv.bill_id and bv.bill_version_id like '%AMD'
        join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
        where b.measure_type in ('AB' , 'SB') and b.session_year < '2015'"""

def amd_bills_query_first():
    return """SELECT b.bill_id, bv.bill_version_id, b.session_year, b.session_num, measure_type, bv.urgency, datediff(bv.bill_version_action_date,sd.start_date) as days_since_start, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee, b.passed
        FROM bill_tbl b
        join bill_version_tbl bv
        on b.bill_id=bv.bill_id and bv.bill_version_id like '%AMD'
        join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
        where b.measure_type in ('AB' , 'SB') and b.session_year < '2015' and bv.bill_version_id in (select max(bill_version_id) as bvid from bill_version_tbl where bill_version_id like '%AMD' group by bill_id)"""

def amd_bills_query_test():
    return """SELECT b.bill_id, bv.bill_version_id, b.session_year, b.session_num, measure_type, bv.urgency, datediff(bv.bill_version_action_date,sd.start_date) as days_since_start, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee, b.passed
        FROM bill_tbl b
        join bill_version_tbl bv
        on b.bill_id=bv.bill_id and bv.bill_version_id like '%AMD'
        join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
        where b.measure_type in ('AB' , 'SB') and b.session_year >= '2015'"""

def amd_n_amendments_query():
    return """select bv1.bill_version_id, bv1.bill_id, count(bv2.bill_version_id) as n_prev_versions from bill_version_tbl bv1
        join bill_version_tbl bv2 on bv1.bill_id=bv2.bill_id
        where bv1.bill_version_id like '%AMD' and bv2.version_num > bv1.version_num
        group by bv1.bill_version_id"""

def amd_text_query():
    return """select bv.bill_version_id, bv.bill_xml from bill_version_tbl bv
        where bv.bill_version_id like '%AMD'"""

def amd_prev_com_query():
    return """SELECT bv.bill_version_id, bv.bill_version_id as bvid2, bsv.SCID
        from bill_version_tbl bv
        left join bill_summary_vote_tbl bsv on bv.bill_id=bsv.bill_id and bv.bill_version_action_date > bsv.vote_date_time
        where bv.bill_version_id like '%AMD' and (bv.bill_id like '%AB%' or bv.bill_id like '%SB%')"""

def amd_authors_query():
    return """SELECT
            bv.bill_id, bv.bill_version_id, bva.session_year, l.author_name, l.legislator_name, l.district, l.party
        FROM
            bill_version_authors_tbl bva
                LEFT JOIN
            legislator_tbl l ON bva.name = l.author_name
                AND l.session_year = bva.session_year
                JOIN
            bill_version_tbl bv ON bv.bill_version_id = bva.bill_version_id
        WHERE
            contribution = 'LEAD_AUTHOR'
                AND bva.bill_version_id LIKE '%AMD'
                AND (bv.bill_id LIKE '%AB%'
                OR bv.bill_id LIKE '%SB%')"""

def final_n_amendments():
    return """SELECT
            t.bill_id, COUNT(t.bill_version_id) AS n_amd
        FROM
            (SELECT
                b.bill_id, amds.bill_version_id
            FROM
                bill_tbl b
            LEFT JOIN (SELECT
                bill_id, bill_version_id
            FROM
                bill_version_tbl bv
            WHERE
                bill_version_id LIKE '%AMD') amds ON b.bill_id = amds.bill_id
                AND b.measure_type IN ('AB' , 'SB')) t
        GROUP BY t.bill_id
"""
