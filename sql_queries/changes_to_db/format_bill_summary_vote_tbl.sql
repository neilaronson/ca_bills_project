ALTER TABLE `capublic`.`bill_summary_vote_tbl` 
ADD COLUMN `SCID` VARCHAR(45) NULL DEFAULT NULL AFTER `session_date`,
ADD COLUMN `SCGID` INT NULL DEFAULT NULL AFTER `SCID`;
UPDATE bill_summary_vote_tbl SET SCID=(CASE
	WHEN location_code='AFLOOR' THEN 'A0'
    WHEN location_code='SFLOOR' THEN 'S0'
    WHEN location_code='CONFCM' THEN 'CONFCM'
	WHEN location_code='CS40' THEN 'S1'
	WHEN location_code='CS40' THEN 'S1'
	WHEN location_code='CS40' THEN 'S1'
	WHEN location_code='CS42' THEN 'S5'
	WHEN location_code='CS42' THEN 'S5'
	WHEN location_code='CS43' THEN 'S7'
	WHEN location_code='CS44' THEN 'S6'
	WHEN location_code='CS45' THEN 'S7'
	WHEN location_code='CS45' THEN 'S7'
	WHEN location_code='CS48' THEN 'S11'
	WHEN location_code='CS50' THEN 'S21'
	WHEN location_code='CS51' THEN 'S16'
	WHEN location_code='CS53' THEN 'S15'
	WHEN location_code='CS54' THEN 'S10'
	WHEN location_code='CS55' THEN 'S17'
	WHEN location_code='CS55' THEN 'S17'
	WHEN location_code='CS56' THEN 'S18'
	WHEN location_code='CS57' THEN 'S10'
	WHEN location_code='CS58' THEN 'S20'
	WHEN location_code='CS59' THEN 'S21'
	WHEN location_code='CS59' THEN 'S21'
	WHEN location_code='CS60' THEN 'S12'
	WHEN location_code='CS61' THEN 'S2'
	WHEN location_code='CS62' THEN 'S4'
	WHEN location_code='CS64' THEN 'S9'
	WHEN location_code='CS66' THEN 'S22'
	WHEN location_code='CS67' THEN 'SE'
	WHEN location_code='CS68' THEN 'SE'
	WHEN location_code='CS69' THEN 'S3'
	WHEN location_code='CS69' THEN 'S3'
	WHEN location_code='CS69' THEN 'S3'
	WHEN location_code='CS70' THEN 'S14'
	WHEN location_code='CS71' THEN 'S8'
	WHEN location_code='CS72' THEN 'S19'
	WHEN location_code='CS73' THEN 'S10'
	WHEN location_code='CS73' THEN 'S10'
	WHEN location_code='CS74' THEN 'S13'
	WHEN location_code='CS74' THEN 'S13'
	WHEN location_code='CX01' THEN 'A3'
	WHEN location_code='CX02' THEN 'A1'
	WHEN location_code='CX03' THEN 'A9'
	WHEN location_code='CX04' THEN 'A10'
	WHEN location_code='CX04' THEN 'A10'
	WHEN location_code='CX05' THEN 'A11'
	WHEN location_code='CX05' THEN 'A11'
	WHEN location_code='CX06' THEN 'A7'
	WHEN location_code='CX06' THEN 'A7'
	WHEN location_code='CX07' THEN 'A12'
	WHEN location_code='CX08' THEN 'A13'
	WHEN location_code='CX09' THEN 'A14'
	WHEN location_code='CX10' THEN 'A15'
	WHEN location_code='CX11' THEN 'A16'
	WHEN location_code='CX12' AND (bill_id LIKE '20132014%' OR bill_id LIKE '20112012%' OR bill_id LIKE '20092010%' OR bill_id LIKE '20072008%')  THEN 'A13'
	WHEN location_code='CX12' AND (bill_id LIKE '20012002%' OR bill_id LIKE '20052006%')  THEN 'AE'
	WHEN location_code='CX12' AND bill_id LIKE '20152016%'  THEN 'AE'
	WHEN location_code='CX13' THEN 'A19'
	WHEN location_code='CX14' THEN 'A20'
	WHEN location_code='CX15' THEN 'A21'
	WHEN location_code='CX16' THEN 'A22'
	WHEN location_code='CX17' THEN 'A24'
	WHEN location_code='CX18' THEN 'A25'
	WHEN location_code='CX19' THEN 'A26'
	WHEN location_code='CX20' THEN 'A27'
	WHEN location_code='CX22' THEN 'A28'
	WHEN location_code='CX23' THEN 'A29'
	WHEN location_code='CX24' THEN 'A31'
	WHEN location_code='CX25' THEN 'A4'
	WHEN location_code='CX27' THEN 'A6'
	WHEN location_code='CX28' THEN 'A17'
	WHEN location_code='CX29' THEN 'A7'
	WHEN location_code='CX30' AND bill_id LIKE '20092010%' THEN 'A9'
	WHEN location_code='CX30' AND bill_id LIKE '20152016%'  THEN 'AE'
	WHEN location_code='CX31' THEN 'A2'
	WHEN location_code='CX32' AND (bill_id LIKE '20092010%' OR bill_id LIKE '20072008%')  THEN 'AE'
	WHEN location_code='CX32' AND bill_id LIKE '20152016%'  THEN 'A23'
	WHEN location_code='CX33' THEN 'A8'
	WHEN location_code='CX33' THEN 'A8'
	WHEN location_code='CX34' THEN 'A18'
	WHEN location_code='CX34' THEN 'A18'
	WHEN location_code='CX35' THEN 'AE'
	WHEN location_code='CX35' THEN 'AE'
	WHEN location_code='CX36' THEN 'A26'
	WHEN location_code='CX37' THEN 'A5'
	WHEN location_code='CX38' THEN 'A30'
    ELSE NULL END);
UPDATE bill_summary_vote_tbl SET SCGID=(CASE
	WHEN SCID='S0' OR SCID='A0' THEN 0
	WHEN location_code='CONFCM' THEN 2
	WHEN location_code='CS40' THEN 1
	WHEN location_code='CS40' THEN 1
	WHEN location_code='CS40' THEN 1
	WHEN location_code='CS42' THEN 3
	WHEN location_code='CS42' THEN 3
	WHEN location_code='CS43' THEN 5
	WHEN location_code='CS44' THEN 4
	WHEN location_code='CS45' THEN 5
	WHEN location_code='CS45' THEN 5
	WHEN location_code='CS48' THEN 5
	WHEN location_code='CS50' THEN 9
	WHEN location_code='CS51' THEN 3
	WHEN location_code='CS53' THEN 5
	WHEN location_code='CS54' THEN 5
	WHEN location_code='CS55' THEN 1
	WHEN location_code='CS55' THEN 1
	WHEN location_code='CS56' THEN 7
	WHEN location_code='CS57' THEN 5
	WHEN location_code='CS58' THEN 5
	WHEN location_code='CS59' THEN 9
	WHEN location_code='CS59' THEN 9
	WHEN location_code='CS60' THEN 6
	WHEN location_code='CS61' THEN 5
	WHEN location_code='CS62' THEN 5
	WHEN location_code='CS64' THEN 1
	WHEN location_code='CS66' THEN 5
	WHEN location_code='CS67' THEN 9
	WHEN location_code='CS68' THEN 6
	WHEN location_code='CS69' THEN 3
	WHEN location_code='CS69' THEN 3
	WHEN location_code='CS69' THEN 3
	WHEN location_code='CS70' THEN 3
	WHEN location_code='CS71' THEN 9
	WHEN location_code='CS72' THEN 8
	WHEN location_code='CS73' THEN 5
	WHEN location_code='CS73' THEN 5
	WHEN location_code='CS74' THEN 6
	WHEN location_code='CS74' THEN 6
	WHEN location_code='CX01' THEN 1
	WHEN location_code='CX02' THEN 5
	WHEN location_code='CX03' THEN 4
	WHEN location_code='CX04' THEN 5
	WHEN location_code='CX04' THEN 5
	WHEN location_code='CX05' THEN 1
	WHEN location_code='CX05' THEN 1
	WHEN location_code='CX06' THEN 5
	WHEN location_code='CX06' THEN 5
	WHEN location_code='CX07' THEN 5
	WHEN location_code='CX08' THEN 6
	WHEN location_code='CX09' THEN 4
	WHEN location_code='CX10' THEN 9
	WHEN location_code='CX11' THEN 6
	WHEN location_code='CX12' AND (bill_id LIKE '20132014%' OR bill_id LIKE '20112012%' OR bill_id LIKE '20092010%' OR bill_id LIKE '20072008%')  THEN 6
	WHEN location_code='CX12' AND (bill_id LIKE '20012002%' OR bill_id LIKE '20052006%')  THEN 8
	WHEN location_code='CX12' AND bill_id LIKE '20152016%'  THEN 9
	WHEN location_code='CX13' THEN 5
	WHEN location_code='CX14' THEN 3
	WHEN location_code='CX15' THEN 5
	WHEN location_code='CX16' THEN 1
	WHEN location_code='CX17' THEN 7
	WHEN location_code='CX18' THEN 8
	WHEN location_code='CX19' THEN 5
	WHEN location_code='CX20' THEN 5
	WHEN location_code='CX22' THEN 9
	WHEN location_code='CX23' THEN 9
	WHEN location_code='CX24' THEN 1
	WHEN location_code='CX25' THEN 5
	WHEN location_code='CX27' THEN 3
	WHEN location_code='CX28' THEN 3
	WHEN location_code='CX29' THEN 5
	WHEN location_code='CX30' AND bill_id LIKE '20092010%' THEN 4
	WHEN location_code='CX30' AND bill_id LIKE '20152016%'  THEN 6
	WHEN location_code='CX31' THEN 6
	WHEN location_code='CX32' AND (bill_id LIKE '20092010%' OR bill_id LIKE '20072008%')  THEN 1
	WHEN location_code='CX32' AND bill_id LIKE '20152016%'  THEN 3
	WHEN location_code='CX33' THEN 3
	WHEN location_code='CX33' THEN 3
	WHEN location_code='CX34' THEN 3
	WHEN location_code='CX34' THEN 3
	WHEN location_code='CX35' THEN 3
	WHEN location_code='CX35' THEN 9
	WHEN location_code='CX36' THEN 5
	WHEN location_code='CX37' THEN 3
	WHEN location_code='CX38' THEN 5
	ELSE NULL END);
