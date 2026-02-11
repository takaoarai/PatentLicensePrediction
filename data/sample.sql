DECLARE start_year INT64 DEFAULT 2000;
DECLARE end_year INT64 DEFAULT 2017;

WHILE start_year <= end_year DO

  EXPORT DATA OPTIONS (
    uri = FORMAT("gs://FILLHERE/patents_%d_*.parquet", start_year),
    format = 'PARQUET',
    overwrite = true
  )
  AS
  WITH filtered_patents AS (
    SELECT
      number,
      id,
      SAFE.PARSE_DATE('%Y-%m-%d', date) AS publication_date,
      type,
      num_claims,
      abstract
    FROM `patents-public-data.patentsview.patent`
    WHERE EXTRACT(YEAR FROM SAFE.PARSE_DATE('%Y-%m-%d', date)) = start_year
  ),

  application_data AS (
    SELECT
      patent_id,
      SAFE.PARSE_DATE('%Y-%m-%d', date) AS application_date
    FROM `patents-public-data.patentsview.application`
  ),

  patent_family_ids AS (
    SELECT
      publication_number,
      family_id
    FROM `patents-public-data.patents.publications`
    WHERE publication_number IN (SELECT number FROM filtered_patents)
  ),
  
  claims_text AS (
    SELECT cl.patent_id, STRING_AGG(cl.text, ' || ' ORDER BY CAST(cl.sequence AS INT64)) AS all_claims_text
    FROM `patents-public-data.patentsview.claim` cl
    JOIN filtered_patents p ON cl.patent_id = p.id
    GROUP BY cl.patent_id
  ),

  cpc_codes AS (
    SELECT cpc.patent_id, STRING_AGG(cpc.group_id, ' | ') AS cpc_codes
    FROM `patents-public-data.patentsview.cpc_current` cpc
    JOIN filtered_patents p ON cpc.patent_id = p.id
    GROUP BY cpc.patent_id
  ),

  applicant_names AS (
    SELECT pa.patent_id, STRING_AGG(COALESCE(a.organization, CONCAT(a.name_first, ' ', a.name_last)), ' | ') AS applicant_names
    FROM `patents-public-data.patentsview.patent_assignee` pa
    JOIN `patents-public-data.patentsview.assignee` a ON a.id = pa.assignee_id
    JOIN filtered_patents p ON pa.patent_id = p.id
    GROUP BY pa.patent_id
  ),

  backward_citations AS (
    SELECT
      bc.patent_id AS publication_number,
      STRING_AGG(bc.citation_id, ' | ') AS backward_citations,
      COUNT(bc.citation_id) AS backward_citation_count
    FROM `patents-public-data.patentsview.uspatentcitation` AS bc
    JOIN filtered_patents p ON p.number = bc.patent_id
    GROUP BY bc.patent_id
  ),

  forward_citations AS (
    SELECT
      fc.citation_id AS publication_number,
      ARRAY_AGG(
        STRUCT(
          fc.patent_id AS cited_patent_id,
          SAFE.PARSE_DATE('%Y-%m-%d', citing_patent.date) AS citation_date
        )
        ORDER BY SAFE.PARSE_DATE('%Y-%m-%d', citing_patent.date)
      ) AS forward_citations_with_dates
    FROM `patents-public-data.patentsview.uspatentcitation` AS fc
    JOIN `patents-public-data.patentsview.patent` citing_patent ON fc.patent_id = citing_patent.number
    WHERE fc.citation_id IN (SELECT number FROM filtered_patents)
    GROUP BY fc.citation_id
  ),
  
  patent_licenses AS (
    SELECT
      d.grant_doc_num AS publication_number,
      STRING_AGG(r.or_name, ' | ') AS licensors,
      STRING_AGG(e.ee_name, ' | ') AS licensees,
      MIN(SAFE.PARSE_DATE('%Y-%m-%d', r.exec_dt)) AS record_date
    FROM `patents-public-data.uspto_oce_assignment.assignment` AS assign
    JOIN `patents-public-data.uspto_oce_assignment.documentid` AS d USING(rf_id)
    JOIN `patents-public-data.uspto_oce_assignment.assignor` AS r USING(rf_id)
    JOIN `patents-public-data.uspto_oce_assignment.assignee` AS e USING(rf_id)
    JOIN filtered_patents p ON p.number = d.grant_doc_num
    WHERE REGEXP_CONTAINS(LOWER(assign.convey_text), r'license|confirmatory license|government license')
    GROUP BY d.grant_doc_num, assign.rf_id
    QUALIFY ROW_NUMBER() OVER(PARTITION BY d.grant_doc_num ORDER BY record_date ASC) = 1
  )

  SELECT
    p.number AS publication_number,
    p.publication_date,
    ad.application_date,
    p.type AS patent_type,
    p.num_claims,
    p.abstract,
    pf.family_id,
    COALESCE(cl.all_claims_text, 'No Claims') AS all_claims_text,
    COALESCE(cc.cpc_codes, 'No CPC Codes') AS cpc_codes,
    COALESCE(ap.applicant_names, 'No Applicants') AS applicant_names,
    COALESCE(bc.backward_citations, 'No Backward Citations') AS backward_citations,
    COALESCE(bc.backward_citation_count, 0) AS backward_citation_count,
    COALESCE(fc.forward_citations_with_dates, []) AS forward_citations_with_dates,
    COALESCE(pl.licensors, 'No Licensor') AS licensors,
    COALESCE(pl.licensees, 'No Licensee') AS licensees,
    pl.record_date AS license_record_date,
    CASE WHEN pl.record_date IS NOT NULL THEN TRUE ELSE FALSE END AS s,
    DATE_DIFF(pl.record_date, ad.application_date, DAY) AS t
  FROM filtered_patents p
  LEFT JOIN application_data ad ON p.id = ad.patent_id
  LEFT JOIN patent_family_ids pf ON p.number = pf.publication_number
  LEFT JOIN claims_text cl ON p.id = cl.patent_id
  LEFT JOIN cpc_codes cc ON p.id = cc.patent_id
  LEFT JOIN applicant_names ap ON p.id = ap.patent_id
  LEFT JOIN backward_citations bc ON p.number = bc.publication_number
  LEFT JOIN forward_citations fc ON p.number = fc.publication_number
  LEFT JOIN patent_licenses pl ON p.number = pl.publication_number;

SET start_year = start_year + 1;

END WHILE;
