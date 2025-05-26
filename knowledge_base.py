'''
This file contains the logical rules used for the LTN training.
'''
# SEPSIS KNOWLEDGE BASE
'''
R1: Patients with LacticAcid > 2.0 should be admitted to the ICU. 
    \/ x (LacticAcid(x) > 2.0 -> ICU(x))
R2: Patients with Tachypnea, Suspected Infection, and Critical Heart Rate, should be admitted to the ICU. 
    \/ x (Tachypnea(x) /\ SuspectedInfection(x) /\ CriticalHeartRate(x) -> ICU(x))
R3: Patients with CRP activity followed by IV Antibiotics and with CRP value > 100 should be admitted to the ICU. 
    \/ x (CRP(x) /\ IVAntibiotics(x) /\ CRPValue(x) > 100 -> ICU(x))
R4: The activity ERTriage should be followed by the activity ERSepsisTriage.
R5: The activity ERSepsisTriage should be followed by IV Antibiotics.
R6: The activity ERSepsisTriage should be followed by the activity IV Liquid.
'''
# BPI2012 KNOWLEDGE BASE
'''
R1: If the Requested Amount is less than 10000, then the application should not be accepted. 
    \/ x (RequestedAmount(x) < 10000 -> ApplicationAccepted(x))
R2: If the Requested Amount is greater than 50000 and less than 100000, then the application should not be accepted. 
    \/ x (RequestedAmount(x) > 50000 /\ RequestedAmount(x) < 100000 -> Not(ApplicationAccepted(x)))
R3: If the resources 10910 and 11169 perform an activity, then the application should not be accepted. 
    \/ x (HasResource(x,10910) /\ HasResource(x,11169) -> Not(ApplicationAccepted(x)))
R4: The activity W_Completeren aanvraag-COMPLETE should be followed by the activity A_ACCEPTED_COMPLETE
R5: The activity W_Valideren aanvraag-COMPLETE should be followed by the activity O_ACCEPTED_COMPLETE
R6: The activity O_SENT_BACK_COMPLETE should be followed by the activity W_Valideren aanvraag-COMPLETE
'''
# BPI2017 KNOWLEDGE BASE
'''
R1: If the Credit Score is greater than 0 and the Requested Amount is less than 20000, then the application should be accepted. 
    \/ x (CreditScore(x) > 0 /\ RequestedAmount(x) < 20000 -> ApplicationAccepted(x))
R2: If there is no offer with Credit Score greater than 0, then the application should not be accepted. 
    \/ x (Not(Offer(x)) /\ CreditScore(x) > 0 -> Not(ApplicationAccepted(x)))
R3: If the Requested Amount is greater than 20000 and the Loan Goal is "Existing loan takeover", then the application should be accepted.   
    \/ x (RequestedAmount(x) > 20000 /\ LoanGoal(x) = "Existing loan takeover" -> ApplicationAccepted(x))
R4: The activity A_SUBMITTED should be followed by the activity A_ACCEPTED.
R5: The activity A_ACCEPTED should be followed by the activity O_CREATE_OFFER.
R6: The activity A_Complete should be followed by the activity W_validate application.
'''
# TRAFFIC FINES KNOWLEDGE BASE
'''
R1: If the payment amount is less than the amount of the fine, then the fine shoul be sent for credit collection. 
    \/x (PaymentAmount(x) < FineAmount(x) -> SendToCreditCollection(x))
R2: If the activity Add penalty occurs, then the fine should be sent for credit collection. 
    \/x (HasActivity(x, "Add penalty") -> SendToCreditCollection(x))
R3: If the amount of the fine is greater than 400, then the fine should be sent for credit collection. 
    \/x (FineAmount(x) > 400 -> SendToCreditCollection(x))
R4: The activity Create Fine should be followed by the activity Send Fine.
R5: The activity Send Fine should be followed by the activity Insert Fine Notification.
R6: The activity Send Fine should be followed by the activity Payment.
'''