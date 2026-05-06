package detection

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"rinha-backend-golang/m/v2/internal/domain"
)

type Normalization struct {
	MaxAmount float64 `json:"max_amount"`
	MaxInstallments float64 `json:"max_installments"`
	AmountVsAvgRatio float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes float64 `json:"max_minutes"`
	MaxKm float64 `json:"max_km"`
	MaxTxCount24h float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
}

type Vectorizer struct {
	norm Normalization
	mccRisk map[string]float64
}

func LoadNormalization(path string) (Normalization, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Normalization{}, fmt.Errorf("failed to read normalization file: %w", err)
	}

	var norm Normalization
	if err := json.Unmarshal(data, &norm); err != nil {
		return Normalization{}, fmt.Errorf("failed to unmarshal normalization data: %w", err)
	}

	return norm, nil
}

func LoadMccRisk(path string) (map[string]float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read MCC risk file: %w", err)
	}

	var mccRisk map[string]float64
	if err := json.Unmarshal(data, &mccRisk); err != nil {
		return nil, fmt.Errorf("decode mcc risk json: %w", err)
	}

	return mccRisk, nil
}

func NewVectorizer(norm Normalization, mccRisk map[string]float64) (*Vectorizer, error) {
	if norm.MaxAmount <= 0 || norm.MaxInstallments <= 0 || norm.MaxMinutes <= 0 || norm.MaxKm <= 0 || norm.MaxTxCount24h <= 0 || norm.MaxMerchantAvgAmount <= 0 {
		return nil, fmt.Errorf("all normalization parameters must be greater than 0")
	}
	if mccRisk == nil {
		return nil, fmt.Errorf("mccRisk map cannot be nil")
	}
	return &Vectorizer{
		norm: norm,
		mccRisk: mccRisk,
	}, nil
}

func (v *Vectorizer) Build(req domain.FraudScoreRequest) ([domain.VectorDimensions]float32, error) {
	var out [domain.VectorDimensions]float32


	reqAt, err := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	if err != nil {
		return out, fmt.Errorf("parse requested_at: %w", err)
	}

	out[0] = clampF32(req.Transaction.Amount / v.norm.MaxAmount)
	out[1] = clampF32(float64(req.Transaction.Installments) / v.norm.MaxInstallments)
	out[2] = clampF32(amountVsAvg(req.Transaction.Amount, req.Customer.AvgAmount, v.norm.AmountVsAvgRatio))
	out[3] = float32(reqAt.UTC().Hour()) / 23.0
	out[4] = float32(toMondayZero(reqAt.UTC().Weekday())) / 6.0

	if req.LastTransaction == nil {
		out[5] = -1
		out[6] = -1
	} else {
		lastAt, parseErr := time.Parse(time.RFC3339, req.LastTransaction.Timestamp)
		if parseErr != nil {
			return out, fmt.Errorf("parse last_transaction timestamp: %w", parseErr)
		}

		minutes := reqAt.Sub(lastAt).Minutes()
		if minutes < 0 {
			minutes = 0
		}
		out[5] = clampF32(minutes / v.norm.MaxMinutes)
		out[6] = clampF32(req.LastTransaction.KmFromCurrent / v.norm.MaxKm)
	}

	out[7] = clampF32(req.Terminal.KmFromHome / v.norm.MaxKm)
	out[8] = clampF32(float64(req.Customer.TxCount24h) / v.norm.MaxTxCount24h)
	if req.Terminal.IsOnline {
		out[9] = 1
	}
	if req.Terminal.CardPresent {
		out[10] = 1
	}
	if isUnknownMerchant(req.Merchant.ID, req.Customer.KnownMerchants) {
		out[11] = 1
	}

	risk, ok := v.mccRisk[req.Merchant.MCC]
	if !ok {
		risk = 0.5
	}
	out[12] = float32(risk)
	out[13] = clampF32(req.Merchant.AvgAmount / v.norm.MaxMerchantAvgAmount)

	return out, nil
}

func (v *Vectorizer) RiskForMCC(mcc string) float64 {
	risk, ok := v.mccRisk[mcc]
	if !ok {
		return 0.5
	}
	return risk
}

func amountVsAvg(amount, avg float64, ratio float64) float64 {
	if avg <= 0 {
		return 1
	}
	return (amount / avg) / ratio
}

func clampF32(value float64) float32 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return float32(value)
}

func toMondayZero(weekday time.Weekday) int {
	return (int(weekday) + 6) % 7
}

func isUnknownMerchant(merchantID string, knownMerchants []string) bool {
	merchantID = strings.TrimSpace(merchantID)
	for _, id := range knownMerchants {
		if merchantID == strings.TrimSpace(id) {
			return false
		}
	}
	return true
}
