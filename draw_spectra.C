void draw_spectra() {
    // Open the ROOT file
    TFile* file = new TFile("jet_spectra.root", "READ");
    if (!file->IsOpen()) {
        printf("Error: Could not open file jet_spectra.root\n");
        return;
    }

    // Get the histograms
    TH1F* hJetPt = (TH1F*)file->Get("hJetPt");
    TH1F* hJetEta = (TH1F*)file->Get("hJetEta");
    
    if (!hJetPt || !hJetEta) {
        printf("Error: Could not retrieve histograms\n");
        return;
    }

    // Create a canvas with two pads
    TCanvas* c1 = new TCanvas("c1", "Jet Spectra", 1200, 500);
    c1->Divide(2,1);

    // Style settings
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    // Draw pT spectrum
    c1->cd(1);
    gPad->SetLogy();
    gPad->SetLeftMargin(0.15);
    gPad->SetBottomMargin(0.15);

    hJetPt->SetLineColor(kBlue);
    hJetPt->SetLineWidth(2);
    hJetPt->SetMarkerStyle(20);
    hJetPt->SetMarkerColor(kBlue);
    hJetPt->GetXaxis()->SetTitle("p_{T} (GeV/c)");
    hJetPt->GetYaxis()->SetTitle("Counts");
    hJetPt->GetXaxis()->SetTitleSize(0.05);
    hJetPt->GetYaxis()->SetTitleSize(0.05);
    hJetPt->Draw("PE");

    // Draw eta spectrum
    c1->cd(2);
    gPad->SetLeftMargin(0.15);
    gPad->SetBottomMargin(0.15);

    hJetEta->SetLineColor(kRed);
    hJetEta->SetLineWidth(2);
    hJetEta->SetMarkerStyle(20);
    hJetEta->SetMarkerColor(kRed);
    hJetEta->GetXaxis()->SetTitle("#eta");
    hJetEta->GetYaxis()->SetTitle("Counts");
    hJetEta->GetXaxis()->SetTitleSize(0.05);
    hJetEta->GetYaxis()->SetTitleSize(0.05);
    hJetEta->Draw("PE");

    // Add legend
    TLegend* leg = new TLegend(0.65, 0.75, 0.85, 0.85);
    leg->AddEntry(hJetPt, "Jet p_{T}", "lp");
    leg->AddEntry(hJetEta, "Jet #eta", "lp");
    leg->SetBorderSize(0);
    leg->Draw();

    // Add text for collision system
    TPaveText* pt = new TPaveText(0.2, 0.85, 0.4, 0.95, "NDC");
    pt->SetBorderSize(0);
    pt->SetFillStyle(0);
    pt->AddText("pp #sqrt{s} = 13 TeV");
    pt->Draw();

    // Save the canvas
    c1->SaveAs("jet_spectra.pdf");
    c1->SaveAs("jet_spectra.png");
} 