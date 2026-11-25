#!/usr/bin/env python
"""Ingest sample RBI regulatory documents for testing."""
import asyncio
import uuid
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, UTC
from app.core.database import get_db_context
from app.models.document import Document, DocumentChunk
from app.services.vector_store_sqlite import SQLiteVectorStoreService
from app.services.embeddings import EmbeddingService
from app.services.document_processor import LegalAwareChunker, MetadataExtractor
from sqlalchemy import select

# Sample RBI KYC Master Direction content
SAMPLE_KYC_DOCUMENT = """
RBI Master Direction - Know Your Customer (KYC) Direction, 2016
Updated as on July 01, 2023

Section 1: Introduction
1.1 These Directions are issued by the Reserve Bank of India (RBI) under Section 35A of the Banking Regulation Act, 1949.
1.2 These Directions shall apply to all Scheduled Commercial Banks (SCBs), Regional Rural Banks (RRBs), Local Area Banks (LABs), Primary (Urban) Co-operative Banks (UCBs), State and Central Co-operative Banks, and all Non-Banking Financial Companies (NBFCs).

Section 2: Customer Acceptance Policy
2.1 No account shall be opened in anonymous or fictitious/benami name(s).
2.2 Banks shall prepare a profile for each new customer based on risk categorization.
2.3 The customer profile shall include information relating to customer's identity, social/financial status, nature of business activity, information about clients' business and their location, etc.

Section 3: Customer Identification Procedure (CIP)
3.1 Banks shall carry out Customer Due Diligence (CDD) at the time of:
    (a) Opening of an account
    (b) Carrying out a transaction above Rs. 50,000
    (c) When there is a suspicion of money laundering or terrorist financing
    (d) When the bank has doubts about the veracity of previously obtained customer identification data

3.2 For individual customers, the following documents shall be obtained:
    (a) Aadhaar Number or proof of possession of Aadhaar
    (b) PAN or Form 60
    (c) Recent photograph
    (d) Officially Valid Document (OVD) for identity and address

3.3 For legal entities, the following shall be obtained:
    (a) Certificate of incorporation
    (b) Memorandum and Articles of Association
    (c) PAN of the entity
    (d) Resolution of Board of Directors for opening account
    (e) Power of Attorney for authorized signatories

Section 4: Risk Management
4.1 Banks shall categorize customers into low, medium, and high risk based on:
    (a) Nature of business activity
    (b) Location of customer
    (c) Mode of payments
    (d) Volume of turnover
    (e) Social and financial status

4.2 High-risk customers shall be subjected to enhanced due diligence (EDD).
4.3 Periodic review of risk categorization shall be carried out at least once in six months.

Section 5: Monitoring of Transactions
5.1 Banks shall monitor transactions in accounts for unusual or suspicious activities.
5.2 Complex, unusually large transactions, and all unusual patterns which have no apparent economic or visible lawful purpose shall be examined.
5.3 Banks shall report suspicious transactions to FIU-IND under the Prevention of Money Laundering Act (PMLA), 2002.

Section 6: KYC Updation
6.1 Periodic updation of KYC shall be carried out:
    (a) High risk customers - at least once in two years
    (b) Medium risk customers - at least once in eight years
    (c) Low risk customers - at least once in ten years

6.2 For minor accounts, KYC shall be updated on attaining majority.
6.3 Fresh KYC shall be required for accounts dormant for more than two years.

Section 7: Central KYC Records Registry (CKYCR)
7.1 All reporting entities shall register with the Central KYC Records Registry.
7.2 KYC records of individual customers shall be uploaded to CKYCR within 10 days of onboarding.
7.3 KYC identifier (KYC ID) shall be used for subsequent verifications.

Section 8: Digital KYC / Video KYC
8.1 Video-based Customer Identification Process (V-CIP) is permitted for individual customers.
8.2 V-CIP shall be carried out by trained officers of the bank.
8.3 The video recording shall be stored securely for at least 5 years.
8.4 Aadhaar-based e-KYC authentication is permitted as per Aadhaar Act, 2016.

Section 9: Recent Amendments (2022-2023)
9.1 August 2022: Clarification on use of Aadhaar for KYC - voluntary use permitted with explicit consent.
9.2 November 2022: Introduction of Video KYC for NBFCs and Payment System Operators.
9.3 January 2023: Guidelines on KYC for Fintech companies and Digital Lending Apps.
9.4 April 2023: Simplified KYC for small accounts (up to Rs. 2 lakh balance) with only Aadhaar and PAN.
9.5 July 2023: Enhanced due diligence requirements for Politically Exposed Persons (PEPs) and their family members.
9.6 July 2023: New provisions for KYC of Legal Entity Identifier (LEI) holders.
9.7 July 2023: Clarification on beneficial ownership threshold reduced from 25% to 10% for high-risk customers.

Section 10: Non-Compliance and Penalties
10.1 Violation of these Directions may attract penal action under Section 35A of the Banking Regulation Act, 1949.
10.2 RBI may impose monetary penalties, restrict business operations, or revoke license for persistent non-compliance.
"""

# Sample Basel III document
SAMPLE_BASEL_DOCUMENT = """
Basel III Capital Regulations - RBI Implementation
Updated as on March 31, 2024

Section 1: Scope of Application
1.1 These guidelines apply to all commercial banks in India (excluding Regional Rural Banks and Local Area Banks).
1.2 Consolidated application to banking groups with overseas subsidiaries.

Section 2: Capital Adequacy Requirements
2.1 Minimum Common Equity Tier 1 (CET1) capital ratio: 8.0% (including Capital Conservation Buffer of 2.5%)
2.2 Minimum Tier 1 capital ratio: 9.5% (including Capital Conservation Buffer)
2.3 Minimum Total Capital ratio: 11.5% (including Capital Conservation Buffer)
2.4 Countercyclical Capital Buffer (CCyB): 0% to 2.5% as determined by RBI

Section 3: Definition of Capital
3.1 Common Equity Tier 1 Capital includes:
    (a) Paid-up equity share capital
    (b) Statutory reserves
    (c) Capital reserves representing surplus
    (d) Other disclosed free reserves
    (e) Revaluation reserves (discounted at 55%)
    (f) Less: Goodwill and other intangible assets
    (g) Less: Deferred tax assets
    (h) Less: Defined benefit pension fund assets

3.2 Additional Tier 1 Capital includes:
    (a) Perpetual Non-Cumulative Preference Shares
    (b) Perpetual debt instruments
    (c) Stock surplus from AT1 instruments

3.3 Tier 2 Capital includes:
    (a) General provisions/loan loss reserves (up to 1.25% of credit risk-weighted assets)
    (b) Revaluation reserves (discounted at 55%)
    (c) Subordinated debt instruments (minimum 5 years original maturity)
    (d) Perpetual cumulative preference shares

Section 4: Credit Risk - Standardised Approach
4.1 Risk weights for claims on Sovereigns:
    (a) 0% for claims on Government of India and RBI
    (b) 20% for claims on other sovereigns with AAA to AA- rating
    (c) 50% for claims on sovereigns with A+ to A- rating
    (d) 100% for claims on sovereigns with BBB+ to BBB- rating
    (e) 150% for claims on sovereigns below BBB- or unrated

4.2 Risk weights for claims on Banks:
    (a) 20% for claims on banks with AAA to AA- rating
    (b) 50% for claims on banks with A+ to A- rating
    (c) 100% for claims on banks with BBB+ to BBB- rating
    (d) 150% for claims on banks below BBB- or unrated

4.3 Risk weights for claims on Corporates:
    (a) 20% for AAA to AA-
    (b) 50% for A+ to A-
    (c) 100% for BBB+ to BBB-
    (d) 150% for below BBB- or unrated

Section 5: Market Risk
5.1 Capital charge for interest rate risk: Standardised Duration Method
5.2 Capital charge for equity position risk: 8% general market risk + 8% specific risk
5.3 Capital charge for foreign exchange risk: 8% open position limit
5.4 Capital charge for commodities risk: 15% simplified approach

Section 6: Operational Risk
6.1 Basic Indicator Approach (BIA): 15% of average gross income over 3 years
6.2 Standardised Approach (SA): 12-18% of gross income by business line
6.3 Advanced Measurement Approach (AMA): Internal models subject to RBI approval

Section 7: Leverage Ratio
7.1 Minimum leverage ratio: 3.5% for all banks
7.2 Exposure measure includes on-balance sheet, derivatives, securities financing, and off-balance sheet items

Section 8: Liquidity Coverage Ratio (LCR)
8.1 Minimum LCR: 100% (phased in from 2015)
8.2 High Quality Liquid Assets (HQLA): Level 1 (unlimited), Level 2A (40% cap), Level 2B (15% cap)
8.3 Net cash outflows over 30-day stress period

Section 9: Net Stable Funding Ratio (NSFR)
9.1 Minimum NSFR: 100%
9.2 Available Stable Funding (ASF) vs Required Stable Funding (RSF)
9.3 ASF factors: 100% for regulatory capital, 90-95% for stable deposits, 50% for less stable deposits
9.4 RSF factors: 0-5% for HQLA, 10-15% for loans to financial institutions, 50-85% for other loans

Section 10: Disclosure Requirements (Pillar 3)
10.1 Quarterly disclosure of capital composition
10.2 Semi-annual disclosure of risk-weighted assets
10.3 Annual disclosure of remuneration policy
10.4 Disclosure of leverage ratio, LCR, NSFR on quarterly basis

Section 11: Recent Updates (2023-2024)
11.1 April 2023: Revised framework for resolution of stressed assets - Prudential Framework for Resolution of Stressed Assets.
11.2 June 2023: Guidelines on Compromise Settlements and Technical Write-offs.
11.3 September 2023: Updated Instructions on Classification and Reporting of Fraud Accounts.
11.4 January 2024: Implementation of Basel III Final Reforms - Revised Standardised Approach for Credit Risk.
11.5 January 2024: Revised Operational Risk Framework - Replacement of AMA with Standardised Approach.
11.6 March 2024: Climate-related Financial Risk Disclosures - Consultation Paper.
"""

async def ingest_sample_data():
    """Ingest sample regulatory documents."""
    embedding_service = EmbeddingService()

    async with get_db_context() as db:
        vector_store = SQLiteVectorStoreService(db)

        # Ingest KYC document
        kyc_doc = Document(
            filename="rbi_kyc_master_direction_2023.pdf",
            original_filename="RBI_KYC_Master_Direction_2023.pdf",
            file_size=len(SAMPLE_KYC_DOCUMENT.encode()),
            content_type="application/pdf",
            document_type="regulatory_guideline",
            rbi_circular_number="RBI/2015-16/35A DBR.AML.BC.No.81/14.01.001/2015-16",
            effective_date=datetime(2016, 2, 25),
            jurisdiction="India",
            tags=["KYC", "AML", "RBI", "Master Direction", "Customer Due Diligence"],
            status="processed",
            chunk_count=0,
            processed_at=datetime.now(UTC)
        )
        db.add(kyc_doc)
        await db.flush()  # Get the ID

        print(f"Created KYC document: {kyc_doc.id}")

        # Chunk KYC document
        kyc_chunks = LegalAwareChunker.chunk_text(SAMPLE_KYC_DOCUMENT, chunk_size=800, overlap=150)
        print(f"KYC chunks: {len(kyc_chunks)}")

        # Prepare chunks with metadata
        kyc_chunks_data = []
        for i, chunk in enumerate(kyc_chunks):
            chunk_dict = {
                "content": chunk["content"],
                "token_count": chunk["token_count"],
                "chunk_index": i,
                "page_number": chunk.get("page_number"),
                "section_title": chunk.get("section_title"),
                "clause_reference": None,
                "metadata": {"source": "KYC_Master_Direction", "section": chunk.get("section_title", "")},
                "bm25_tokens": chunk["content"].lower().split()[:100]  # First 100 tokens for BM25
            }
            kyc_chunks_data.append(chunk_dict)

        # Generate embeddings
        kyc_embeddings = await embedding_service.embed_documents([c["content"] for c in kyc_chunks_data])

        # Add to vector store
        await vector_store.add_document_chunks(kyc_chunks_data, kyc_embeddings, str(kyc_doc.id))

        # Update document chunk count
        kyc_doc.chunk_count = len(kyc_chunks_data)

        # Ingest Basel document
        basel_doc = Document(
            filename="basel_iii_rbi_implementation_2024.pdf",
            original_filename="Basel_III_RBI_Implementation_2024.pdf",
            file_size=len(SAMPLE_BASEL_DOCUMENT.encode()),
            content_type="application/pdf",
            document_type="regulatory_guideline",
            rbi_circular_number="DBR.No.BP.BC.83/21.06.201/2015-16",
            effective_date=datetime(2015, 3, 31),
            jurisdiction="India",
            tags=["Basel III", "Capital Adequacy", "Risk Management", "LCR", "NSFR", "Leverage Ratio"],
            status="processed",
            chunk_count=0,
            processed_at=datetime.now(UTC)
        )
        db.add(basel_doc)
        await db.flush()

        print(f"Created Basel document: {basel_doc.id}")

        # Chunk Basel document
        basel_chunks = LegalAwareChunker.chunk_text(SAMPLE_BASEL_DOCUMENT, chunk_size=800, overlap=150)
        print(f"Basel chunks: {len(basel_chunks)}")

        basel_chunks_data = []
        for i, chunk in enumerate(basel_chunks):
            chunk_dict = {
                "content": chunk["content"],
                "token_count": chunk["token_count"],
                "chunk_index": i,
                "page_number": chunk.get("page_number"),
                "section_title": chunk.get("section_title"),
                "clause_reference": None,
                "metadata": {"source": "Basel_III_Implementation", "section": chunk.get("section_title", "")},
                "bm25_tokens": chunk["content"].lower().split()[:100]
            }
            basel_chunks_data.append(chunk_dict)

        basel_embeddings = await embedding_service.embed_documents([c["content"] for c in basel_chunks_data])

        await vector_store.add_document_chunks(basel_chunks_data, basel_embeddings, str(basel_doc.id))

        basel_doc.chunk_count = len(basel_chunks_data)

        await db.commit()
        print("Sample data ingestion complete!")
        print(f"KYC document chunks: {kyc_doc.chunk_count}")
        print(f"Basel document chunks: {basel_doc.chunk_count}")

if __name__ == "__main__":
    asyncio.run(ingest_sample_data())