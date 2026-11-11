# Ingatan eval report

Date: 2026-11-11  
Result: 40 scenarios: 40 pass, 0 fail, 0 xfail, 0 xpass, 0 skipped  
Network attempts during the run: 0

## Metrics

| Metric | Value | Target |
|---|---|---|
| recall@5 | 100.0% (45 checks) | over 80% |
| stale rate | 0.0% (63 recall checks) | under 5% |
| leak rate | 0.0% (0 of 86 results) | 0 |
| tokens per brief | mean 65, max 278 (19 briefs) | about 1,000 |

Definitions are in `evals/README.md`.

## Scenarios

| Scenario | Category | Source | Status | First problem |
|---|---|---|---|---|
| brief-01-check-first-terpisah | brief | pattern | PASS |  |
| brief-02-confirm-keluar-dari-check-first | brief | pattern | PASS |  |
| brief-03-anggaran-token-dan-pemangkasan | brief | pattern | PASS |  |
| brief-04-urutan-bagian-global-bisnis-agen | brief | pattern | PASS |  |
| conflict-01-fakta-mirip-ditandai | conflict | pattern | PASS |  |
| conflict-02-confirm-dengan-id-menyelesaikan | conflict | pattern | PASS |  |
| forget-01-fakta-agen-sendiri | forget | pattern | PASS |  |
| forget-02-fakta-bisnis-tidak-bisa-dihapus-agen | forget | pattern | PASS |  |
| forget-03-fakta-bisnis-dihapus-oleh-user | forget | pattern | PASS |  |
| inflection-01-kirim-vs-pengiriman | inflection | pattern | PASS |  |
| inflection-02-pesan-vs-pemesanan | inflection | pattern | PASS |  |
| inflection-03-tagih-vs-penagihan | inflection | pattern | PASS |  |
| inflection-04-terima-vs-penerimaan | inflection | pattern | PASS |  |
| inflection-05-sewa-vs-penyewaan | inflection | pattern | PASS |  |
| inflection-06-sinonim-diskon-vs-potongan | inflection | pattern | PASS |  |
| inflection-07-bentuk-turunan-beda-dari-akar-di-keywords | inflection | pattern | PASS |  |
| inflection-08-dikirim-vs-akar-kirim | inflection | pattern | PASS |  |
| leak-agent-01-catatan-pribadi-teks-persis | leak-agent | pattern | PASS |  |
| leak-agent-02-brief-tidak-memuat-catatan-agen-lain | leak-agent | pattern | PASS |  |
| leak-agent-03-supersede-fakta-agen-lain-ditolak | leak-agent | pattern | PASS |  |
| leak-agent-04-forget-fakta-agen-lain-ditolak | leak-agent | pattern | PASS |  |
| leak-business-01-fakta-bisnis-lain-tidak-terlihat | leak-business | pattern | PASS |  |
| leak-business-02-fakta-global-terlihat-di-semua-bisnis | leak-business | pattern | PASS |  |
| leak-business-03-brief-dan-hitungan-proposal-bisnis-lain | leak-business | pattern | PASS |  |
| leak-business-04-nama-agen-sama-di-bisnis-berbeda | leak-business | pattern | PASS |  |
| partial-01-langgan | partial-word | pattern | PASS |  |
| partial-02-blokir | partial-word | pattern | PASS |  |
| partial-03-ekspedi | partial-word | pattern | PASS |  |
| proposal-01-pengusul-tidak-melihat-usulannya | proposal | pattern | PASS |  |
| proposal-02-setelah-disetujui-terlihat-agen-lain | proposal | pattern | PASS |  |
| proposal-03-ditolak-tidak-pernah-muncul | proposal | pattern | PASS |  |
| proposal-04-supersede-fakta-bisnis-menunggu-persetujuan | proposal | pattern | PASS |  |
| proposal-05-supersede-ditolak-fakta-kembali-valid | proposal | pattern | PASS |  |
| stale-01-ongkir-lewat-31-hari | stale | pattern | PASS |  |
| stale-02-jadwal-review-eksplisit | stale | pattern | PASS |  |
| stale-03-confirm-valid-lagi | stale | pattern | PASS |  |
| stale-04-fakta-stabil-tidak-basi | stale | pattern | PASS |  |
| supersede-01-harga-lama-tidak-muncul | supersede | pattern | PASS |  |
| supersede-02-rantai-tiga-versi | supersede | pattern | PASS |  |
| supersede-03-riwayat-dengan-include-closed | supersede | pattern | PASS |  |

## Expected failures (xfail)

None.
## Briefs measured

| Scenario | Caller | Tokens | Budget |
|---|---|---|---|
| brief-01-check-first-terpisah | tokoku/asisten | 137 | 1000 |
| brief-02-confirm-keluar-dari-check-first | tokoku/asisten | 95 | 1000 |
| brief-02-confirm-keluar-dari-check-first | tokoku/asisten | 45 | 1000 |
| brief-03-anggaran-token-dan-pemangkasan | tokoku/asisten | 278 | 300 |
| brief-04-urutan-bagian-global-bisnis-agen | tokoku/seo-agent | 99 | 1000 |
| forget-01-fakta-agen-sendiri | tokoku/asisten | 19 | 1000 |
| forget-03-fakta-bisnis-dihapus-oleh-user | tokoku/seo-agent | 20 | 1000 |
| leak-agent-02-brief-tidak-memuat-catatan-agen-lain | tokoku/seo-agent | 48 | 1000 |
| leak-agent-02-brief-tidak-memuat-catatan-agen-lain | tokoku/asisten | 93 | 1000 |
| leak-business-02-fakta-global-terlihat-di-semua-bisnis | kopisenja/asisten | 49 | 1000 |
| leak-business-03-brief-dan-hitungan-proposal-bisnis-lain | kopisenja/asisten | 45 | 1000 |
| leak-business-03-brief-dan-hitungan-proposal-bisnis-lain | tokoku/asisten | 70 | 1000 |
| leak-business-04-nama-agen-sama-di-bisnis-berbeda | kopisenja/asisten | 45 | 1000 |
| proposal-01-pengusul-tidak-melihat-usulannya | tokoku/asisten | 19 | 1000 |
| proposal-01-pengusul-tidak-melihat-usulannya | tokoku/seo-agent | 20 | 1000 |
| proposal-02-setelah-disetujui-terlihat-agen-lain | tokoku/seo-agent | 47 | 1000 |
| proposal-03-ditolak-tidak-pernah-muncul | tokoku/seo-agent | 20 | 1000 |
| stale-04-fakta-stabil-tidak-basi | tokoku/asisten | 42 | 1000 |
| supersede-02-rantai-tiga-versi | tokoku/penulis | 44 | 1000 |
