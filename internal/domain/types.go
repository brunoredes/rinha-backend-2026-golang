package domain

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

const VectorDimensions = 14

type FraudScoreRequest struct {
	ID string `json:"id"`
	Transaction Transaction `json:"transaction"`
	Customer Customer `json:"customer"`
	Merchant Merchant `json:"merchant"`
	Terminal Terminal `json:"terminal"`
	LastTransaction *LastTransaction `json:"last_transaction"`
}

type Transaction struct {
	Amount float64 `json:"amount"`
	Installments int `json:"installments"`
	RequestedAt string `json:"requested_at"`
}

type Customer struct {
	AvgAmount float64 `json:"avg_amount"`
	TxCount24h int `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID string `json:"id"`
	MCC string `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline bool `json:"is_online"`
	CardPresent bool `json:"card_present"`
	KmFromHome float64 `json:"km_from_home"`
}

type LastTransaction struct {
	Timestamp string `json:"timestamp"`
	KmFromCurrent float64 `json:"km_from_current"`
}

type FraudScoreResponse struct {
	Approved bool `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

func (r *FraudScoreRequest) Validate() error {
	if strings.TrimSpace(r.ID) == "" {
		return errors.New("id is required")
	}

	if r.Transaction.Amount <= 0 {
		return errors.New("amount must be greater than 0")
	}

	if r.Transaction.Installments < 0 {
		return errors.New("installments must be higher than 0")
	}

	if _, err := time.Parse(time.RFC3339, r.Transaction.RequestedAt); err != nil {
		return fmt.Errorf("invalid requested_at format: %v", err)
	}

	if r.Customer.AvgAmount < 0 {
		return errors.New("avg_amount must be non-negative")
	}

	if strings.TrimSpace(r.Merchant.ID) == "" {
		return errors.New("merchant id is required")
	}

	if strings.TrimSpace(r.Merchant.MCC) == "" {
		return errors.New("merchant mcc is required")
	}

	if r.Merchant.AvgAmount < 0 {
		return errors.New("merchant avg_amount must be non-negative")
	}

	if r.Terminal.KmFromHome < 0 {
		return errors.New("km_from_home must be non-negative")
	}

	if r.LastTransaction == nil {
		return nil // last_transaction is optional
	}

	if _, err := time.Parse(time.RFC3339, r.LastTransaction.Timestamp); err != nil {
		return fmt.Errorf("invalid last_transaction timestamp format: %w", err)
	}

	if r.LastTransaction.KmFromCurrent < 0 {
		return errors.New("km_from_current must be non-negative")
	}

	return nil
}
